import argparse

import numpy as np
import pymysql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估以图搜图 Top1/Top5 准确率（MySQL）")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评估查询集")
    parser.add_argument("--target-top1", type=float, default=0.98, help="目标 Top1 准确率")

    parser.add_argument("--mysql-host", type=str, default="127.0.0.1", help="MySQL 主机")
    parser.add_argument("--mysql-port", type=int, default=3306, help="MySQL 端口")
    parser.add_argument("--mysql-user", type=str, default="root", help="MySQL 用户名")
    parser.add_argument("--mysql-password", type=str, default="", help="MySQL 密码")
    parser.add_argument("--mysql-database", type=str, default="image_search", help="MySQL 数据库名")
    parser.add_argument("--mysql-table", type=str, default="image_features", help="MySQL 表名")
    return parser.parse_args()


def get_conn(args: argparse.Namespace):
    return pymysql.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
        charset="utf8mb4",
    )


def load_features(conn, table_name: str):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT image_path, label, split_name, feature, dim FROM `{table_name}`")
        rows = cursor.fetchall()

    data = []
    for image_path, label, split_name, feat_blob, dim in rows:
        feat = np.frombuffer(feat_blob, dtype=np.float32, count=dim)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        data.append((image_path, label, split_name, feat))
    return data


def main() -> None:
    args = parse_args()
    conn = get_conn(args)
    data = load_features(conn, args.mysql_table)
    conn.close()

    if not data:
        raise SystemExit("No features found in MySQL table.")

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
        sim[qi] = -1.0
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
