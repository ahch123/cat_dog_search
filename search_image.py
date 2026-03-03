import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymysql
import torch
from PIL import Image
from torchvision import models, transforms


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以图搜图（Top5，MySQL）")
    parser.add_argument("--query-image", type=str, default="", help="查询图片路径（可选）")
    parser.add_argument(
        "--query-dir",
        type=str,
        default="D:\software\datasets\cat_verify",
        help="查询图片目录（可选）。不传 --query-image 时可用目录+序号快速选图。",
    )
    parser.add_argument("--query-index", type=int, default=1, help="在目录列表中的序号（从 1 开始）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="训练权重")
    parser.add_argument("--topk", type=int, default=5, help="返回最相似图片数量")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像大小")
    parser.add_argument(
        "--open-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="检索完成后是否自动打开返回图片（默认开启）",
    )

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


def list_query_images(query_dir: Path) -> list[Path]:
    if not query_dir.exists() or not query_dir.is_dir():
        raise SystemExit(f"Query directory not found: {query_dir}")

    images = [p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    images.sort(key=lambda p: p.name)
    if not images:
        raise SystemExit(f"No supported images found in directory: {query_dir}")
    return images


def resolve_query_image(args: argparse.Namespace) -> Path:
    if args.query_image:
        query_path = Path(args.query_image)
        if not query_path.exists():
            raise SystemExit(f"Query image not found: {query_path}")
        return query_path

    if args.query_dir:
        images = list_query_images(Path(args.query_dir))
        print("Query image list:")
        for idx, path in enumerate(images, start=1):
            print(f"{idx}. {path.name}")

        if args.query_index <= 0 or args.query_index > len(images):
            raise SystemExit(f"query-index out of range: {args.query_index} (1-{len(images)})")
        chosen = images[args.query_index - 1]
        print(f"Selected by index: {chosen}")
        return chosen

    user_input = input("请输入查询图片路径（直接拖拽图片到终端也可以）: ").strip().strip('"').strip("'")
    if not user_input:
        raise SystemExit("No query image provided.")
    query_path = Path(user_input)
    if not query_path.exists():
        raise SystemExit(f"Query image not found: {query_path}")
    return query_path


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


def open_image(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            return
        subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Warning: unable to open image {path}: {exc}")


def open_top_results(top_results: list[tuple[float, str, str, str]]) -> None:
    print("\nOpening top result images...")
    for _, image_path, _, _ in top_results:
        path = Path(image_path)
        if path.exists():
            open_image(path)
        else:
            print(f"Warning: result image not found on disk: {path}")


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise SystemExit("topk must be > 0")

    query_path = resolve_query_image(args)

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

    if args.open_results:
        open_top_results(top_results)


if __name__ == "__main__":
    main()