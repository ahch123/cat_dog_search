import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以图搜图（Top5）")
    parser.add_argument("--query-image", type=str, required=True, help="查询图片路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="训练权重")
    parser.add_argument("--db-path", type=str, default="image_features.db", help="特征数据库路径")
    parser.add_argument("--topk", type=int, default=5, help="返回最相似图片数量")
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


def extract_feature(
    model: torch.nn.Module, image_path: Path, transform: transforms.Compose, device: torch.device
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze(0)
        feat = torch.nn.functional.normalize(feat, p=2, dim=0)
    return feat.cpu().numpy().astype(np.float32)


def load_features(db_path: str) -> list[tuple[str, str, str, np.ndarray]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT image_path, label, split, feature, dim FROM image_features")
    rows = []
    for image_path, label, split, feature_blob, dim in cursor:
        feat = np.frombuffer(feature_blob, dtype=np.float32, count=dim)
        rows.append((image_path, label, split, feat))
    conn.close()
    return rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise SystemExit("topk must be > 0")

    if not os.path.exists(args.db_path):
        raise SystemExit(f"Database not found: {args.db_path}")

    query_path = Path(args.query_image)
    if not query_path.exists():
        raise SystemExit(f"Query image not found: {query_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_model(args.checkpoint, device)
    transform = build_transform(args.image_size)

    query_feat = extract_feature(model, query_path, transform, device)
    rows = load_features(args.db_path)
    if not rows:
        raise SystemExit("No features found in database.")

    scored = []
    for image_path, label, split, feat in rows:
        score = cosine_similarity(query_feat, feat)
        scored.append((score, image_path, label, split))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[: args.topk]

    print(f"Query: {query_path}")
    print("Top matches:")
    for rank, (score, image_path, label, split) in enumerate(top_results, start=1):
        print(f"{rank}. score={score:.4f} | label={label} | split={split} | path={image_path}")


if __name__ == "__main__":
    main()
