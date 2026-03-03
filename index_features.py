import argparse
import os
from pathlib import Path

import numpy as np
import pymysql
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取图像特征并保存到 MySQL 数据库")
    parser.add_argument("--data-dir", type=str, default="data", help="含 train/val 的数据目录")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="训练得到的权重")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像大小")

    parser.add_argument("--mysql-host", type=str, default="127.0.0.1", help="MySQL 主机")
    parser.add_argument("--mysql-port", type=int, default=3306, help="MySQL 端口")
    parser.add_argument("--mysql-user", type=str, default="root", help="MySQL 用户名")
    parser.add_argument("--mysql-password", type=str, default="", help="MySQL 密码")
    parser.add_argument("--mysql-database", type=str, default="image_search", help="MySQL 数据库名")
    parser.add_argument("--mysql-table", type=str, default="image_features", help="MySQL 表名")
    return parser.parse_args()


def build_feature_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2),
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def iter_images(data_dir: Path) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for label in ["cat", "dog"]:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for path in label_dir.iterdir():
                if path.is_file() and path.suffix.lower() in SUPPORTED:
                    items.append((path, label, split))
    return items


def get_conn(args: argparse.Namespace):
    return pymysql.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
        charset="utf8mb4",
        autocommit=False,
    )


def init_table(conn, table_name: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                image_path VARCHAR(1024) NOT NULL UNIQUE,
                label VARCHAR(32) NOT NULL,
                split_name VARCHAR(32) NOT NULL,
                feature LONGBLOB NOT NULL,
                dim INT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    conn.commit()


def upsert_feature(conn, table_name: str, image_path: str, label: str, split_name: str, feature: np.ndarray) -> None:
    feature = feature.astype(np.float32)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO `{table_name}` (image_path, label, split_name, feature, dim)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                label = VALUES(label),
                split_name = VALUES(split_name),
                feature = VALUES(feature),
                dim = VALUES(dim)
            """,
            (image_path, label, split_name, feature.tobytes(), feature.shape[0]),
        )


def extract_feature(
    model: torch.nn.Module, image_path: Path, transform: transforms.Compose, device: torch.device
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze(0)
        feat = torch.nn.functional.normalize(feat, p=2, dim=0)
    return feat.cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_feature_model(args.checkpoint, device)
    transform = build_transform(args.image_size)

    data_items = iter_images(Path(args.data_dir))
    if not data_items:
        raise SystemExit("No images found in data directory.")

    conn = get_conn(args)
    init_table(conn, args.mysql_table)

    for image_path, label, split_name in tqdm(data_items, desc="Indexing"):
        feature = extract_feature(model, image_path, transform, device)
        upsert_feature(conn, args.mysql_table, os.path.abspath(str(image_path)), label, split_name, feature)

    conn.commit()
    conn.close()
    print(f"Indexed {len(data_items)} images into MySQL {args.mysql_database}.{args.mysql_table}")


if __name__ == "__main__":
    main()
