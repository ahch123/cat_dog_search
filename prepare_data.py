import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按文件名前缀拆分猫狗数据集")
    parser.add_argument("--source-dir", type=str, required=True, help="原始图片目录")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def collect_images(source_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]


def split_images(images: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    rng.shuffle(images)
    split_index = int(len(images) * train_ratio)
    return images[:split_index], images[split_index:]


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "train_cat": root / "train" / "cat",
        "train_dog": root / "train" / "dog",
        "val_cat": root / "val" / "cat",
        "val_dog": root / "val" / "dog",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def target_subdir(filename: str) -> str | None:
    if filename.startswith("0"):
        return "cat"
    if filename.startswith("1"):
        return "dog"
    return None


def copy_images(images: list[Path], dest_root: Path, split: str) -> tuple[int, int, int]:
    cat_count = 0
    dog_count = 0
    skipped = 0
    for image_path in images:
        label = target_subdir(image_path.name)
        if label is None:
            skipped += 1
            continue
        dest_dir = dest_root / split / label
        shutil.copy2(image_path, dest_dir / image_path.name)
        if label == "cat":
            cat_count += 1
        else:
            dog_count += 1
    return cat_count, dog_count, skipped


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    images = collect_images(source_dir)
    if not images:
        raise SystemExit("No supported image files found.")

    train_images, val_images = split_images(images, args.train_ratio, args.seed)
    output_root = Path(args.output_dir).expanduser().resolve()
    ensure_dirs(output_root)

    train_cat, train_dog, train_skipped = copy_images(train_images, output_root, "train")
    val_cat, val_dog, val_skipped = copy_images(val_images, output_root, "val")

    total = len(images)
    skipped = train_skipped + val_skipped
    print("Split complete")
    print(f"Total images: {total}")
    print(f"Train: {len(train_images)} (cat={train_cat}, dog={train_dog}, skipped={train_skipped})")
    print(f"Val: {len(val_images)} (cat={val_cat}, dog={val_dog}, skipped={val_skipped})")
    if skipped:
        print("Skipped files are those without 0/1 prefix in filename.")


if __name__ == "__main__":
    main()
