import argparse
from pathlib import Path

import numpy as np
import pymysql
import torch
from PIL import Image
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以图搜图（Top5，MySQL）")
    parser.add_argument("--query-image", type=str, required=True, help="查询图片路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="训练权重")
    parser.add_argument("--topk", type=int, default=5, help="返回最相似图片数量")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像大小")

    parser.add_argument("--mysql-host", type=str, default="192.168.2.36", help="MySQL 主机")
    parser.add_argument("--mysql-port", type=int, default=3306, help="MySQL 端口")
    parser.add_argument("--mysql-user", type=str, default="root", help="MySQL 用户名")
    parser.add_argument("--mysql-password", type=str, default="666666", help="MySQL 密码")
    parser.add_argument("--mysql-database", type=str, default="image_search", help="MySQL 数据库名")
    parser.add_argument("--mysql-table", type=str, default="image_features", help="MySQL 表名")
    return parser.parse_args()


def safe_torch_load(checkpoint_path: str, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_feature_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2),
    )

    ckpt = safe_torch_load(checkpoint_path, device)
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


def extract_feature(
    model: torch.nn.Module, image_path: Path, transform: transforms.Compose, device: torch.device
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze(0)
        feat = torch.nn.functional.normalize(feat, p=2, dim=0)
    return feat.cpu().numpy().astype(np.float32)


def get_conn(args: argparse.Namespace):
    return pymysql.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
        charset="utf8mb4",
    )


def load_features(conn, table_name: str) -> list[tuple[str, str, str, np.ndarray]]:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT image_path, label, split_name, feature, dim FROM `{table_name}`")
        rows = cursor.fetchall()

    data = []
    for image_path, label, split_name, feature_blob, dim in rows:
        feat = np.frombuffer(feature_blob, dtype=np.float32, count=dim)
        data.append((image_path, label, split_name, feat))
    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise SystemExit("topk must be > 0")

    query_path = Path(args.query_image)
    if not query_path.exists():
        raise SystemExit(f"Query image not found: {query_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_model(args.checkpoint, device)
    transform = build_transform(args.image_size)

    query_feat = extract_feature(model, query_path, transform, device)

    conn = get_conn(args)
    rows = load_features(conn, args.mysql_table)
    conn.close()
    if not rows:
        raise SystemExit("No features found in MySQL table.")

    scored = []
    for image_path, label, split_name, feat in rows:
        score = cosine_similarity(query_feat, feat)
        scored.append((score, image_path, label, split_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[: min(args.topk, 5)]

    print(f"Query: {query_path}")
    print("Top matches:")
    for rank, (score, image_path, label, split_name) in enumerate(top_results, start=1):
        print(f"{rank}. score={score:.4f} | label={label} | split={split_name} | path={image_path}")


if __name__ == "__main__":
    main()