import argparse
import os
import random
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


@dataclass
class Config:
    data_dir: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    output_dir: str
    patience: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(os.path.join(config.data_dir, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(config.data_dir, "val"), val_transforms)
    train_dataset = datasets.ImageFolder(os.path.join(config.data_dir, "D:\software\datasets\my_data\\train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(config.data_dir, "D:\software\datasets\my_data\\val"), val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total if total else 0.0
    return val_loss, val_acc


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="猫狗二分类（CNN）")
    parser.add_argument("--data-dir", type=str, default="data", help="数据集目录")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像大小")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正则")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心轮数")
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(config.output_dir, exist_ok=True)
    best_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": config.image_size,
                    "best_acc": best_acc,
                    "arch": "resnet18",
                    "class_to_idx": train_loader.dataset.class_to_idx,
                },
                os.path.join(config.output_dir, "best.pt"),
            )
            print(f"Saved best model with acc={best_acc:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
