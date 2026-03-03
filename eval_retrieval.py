import argparse
import sqlite3

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估以图搜图 Top1/Top5 准确率")
    parser.add_argument("--db-path", type=str, default="image_features.db", help="特征数据库路径")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评估查询集")
    parser.add_argument("--target-top1", type=float, default=0.98, help="目标 Top1 准确率")
    return parser.parse_args()


def load_features(db_path: str):
    conn = sqlite3.connect(db_path)
    rows = list(conn.execute("SELECT image_path, label, split, feature, dim FROM image_features"))
    conn.close()

    data = []
    for image_path, label, split, feat_blob, dim in rows:
        feat = np.frombuffer(feat_blob, dtype=np.float32, count=dim)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        data.append((image_path, label, split, feat))
    return data


def main() -> None:
    args = parse_args()
    data = load_features(args.db_path)
    if not data:
        raise SystemExit("No features found in DB.")

    feats = np.stack([d[3] for d in data], axis=0)
    labels = [d[1] for d in data]
    splits = [d[2] for d in data]

    query_indices = [i for i, s in enumerate(splits) if s == args.split]
    if not query_indices:
        raise SystemExit(f"No query images in split={args.split}")

    top1_hit = 0
    top5_hit = 0

    for qi in query_indices:
        sim = feats @ feats[qi]
        sim[qi] = -1.0  # exclude self
        top5_idx = np.argsort(-sim)[:5]

        if labels[top5_idx[0]] == labels[qi]:
            top1_hit += 1
        if any(labels[idx] == labels[qi] for idx in top5_idx):
            top5_hit += 1

    total = len(query_indices)
    top1 = top1_hit / total
    top5 = top5_hit / total

    print(f"Queries: {total}")
    print(f"Top1 accuracy: {top1:.4f}")
    print(f"Top5 accuracy: {top5:.4f}")
    if top1 >= args.target_top1:
        print(f"Target met: Top1 >= {args.target_top1:.2f}")
    else:
        print(f"Target not met yet: Top1 < {args.target_top1:.2f}")


if __name__ == "__main__":
    main()
