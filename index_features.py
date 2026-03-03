import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取图像特征并保存到 SQLite 数据库")
    parser.add_argument("--data-dir", type=str, default="D:\software\datasets\my_data", help="含 train/val 的数据目录")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="训练得到的权重")
    parser.add_argument("--db-path", type=str, default="image_features.db", help="SQLite 数据库路径")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像大小")
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


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            label TEXT,
            split TEXT,
            feature BLOB,
            dim INTEGER
        )
        """
    )
    conn.commit()


def upsert_feature(
    conn: sqlite3.Connection, image_path: str, label: str, split: str, feature: np.ndarray
) -> None:
    feature = feature.astype(np.float32)
    conn.execute(
        """
        INSERT INTO image_features (image_path, label, split, feature, dim)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(image_path) DO UPDATE SET
            label=excluded.label,
            split=excluded.split,
            feature=excluded.feature,
            dim=excluded.dim
        """,
        (image_path, label, split, feature.tobytes(), feature.shape[0]),
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

    conn = sqlite3.connect(args.db_path)
    init_db(conn)

    for image_path, label, split in tqdm(data_items, desc="Indexing"):
        feature = extract_feature(model, image_path, transform, device)
        upsert_feature(conn, os.path.abspath(str(image_path)), label, split, feature)

    conn.commit()
    conn.close()
    print(f"Indexed {len(data_items)} images into {args.db_path}")


if __name__ == "__main__":
    main()
