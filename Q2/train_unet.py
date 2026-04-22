from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import create_dataloaders


NUM_CLASSES = 23


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.head(x)


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)

        present = target_one_hot.sum(dim=dims) > 0
        if torch.any(present):
            dice_loss = 1.0 - dice[present].mean()
        else:
            dice_loss = 1.0 - dice.mean()

        return 0.6 * ce_loss + 0.4 * dice_loss


def fast_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    valid = (targets >= 0) & (targets < num_classes)
    encoded = num_classes * targets[valid] + predictions[valid]
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_class_weights(
    data_loader: torch.utils.data.DataLoader,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)

    for _, masks in data_loader:
        counts += torch.bincount(masks.view(-1), minlength=num_classes).double()

    weights = torch.zeros_like(counts)
    present = counts > 0
    if torch.any(present):
        median_count = counts[present].median()
        weights[present] = torch.sqrt(median_count / counts[present])
        weights[present] = torch.clamp(weights[present], min=0.5, max=6.0)
    return weights.float()


def compute_miou_mdice(confusion_matrix: torch.Tensor) -> Tuple[float, float]:
    confusion_matrix = confusion_matrix.float()
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    present = confusion_matrix.sum(dim=1) > 0
    if not torch.any(present):
        return 0.0, 0.0

    miou = iou[present].mean().item()
    mdice = dice[present].mean().item()
    return miou, mdice


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    processed_items = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(data_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            predictions = logits.argmax(dim=1)
            confusion_matrix += fast_confusion_matrix(
                predictions.cpu(),
                masks.cpu(),
                num_classes=num_classes,
            )
            total_loss += loss.item() * images.size(0)
            processed_items += images.size(0)
            if max_batches is not None and batch_index >= max_batches:
                break

    avg_loss = total_loss / processed_items
    miou, mdice = compute_miou_mdice(confusion_matrix)
    return {
        "loss": avg_loss,
        "miou": miou,
        "mdice": mdice,
    }


def save_metrics_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "train_miou", "train_mdice", "test_loss", "test_miou", "test_mdice"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_svg_plot(values: List[float], title: str, y_label: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 900
    height = 520
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    epochs = list(range(1, len(values) + 1))
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        min_value -= 0.05
        max_value += 0.05

    value_pad = (max_value - min_value) * 0.1
    min_value -= value_pad
    max_value += value_pad

    def x_coord(epoch: int) -> float:
        if len(epochs) == 1:
            return margin_left + plot_width / 2
        return margin_left + (epoch - 1) * plot_width / (len(epochs) - 1)

    def y_coord(value: float) -> float:
        return margin_top + (max_value - value) * plot_height / (max_value - min_value)

    polyline = " ".join(f"{x_coord(epoch):.2f},{y_coord(value):.2f}" for epoch, value in zip(epochs, values))

    y_ticks = []
    for i in range(6):
        value = min_value + i * (max_value - min_value) / 5
        y_ticks.append(value)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf2"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="2"/>',
    ]

    for tick_value in y_ticks:
        y = y_coord(tick_value)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{tick_value:.4f}</text>'
        )

    for epoch in epochs:
        x = x_coord(epoch)
        svg_lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{epoch}</text>'
        )

    svg_lines.append(f'<polyline fill="none" stroke="#dc2626" stroke-width="4" points="{polyline}"/>')

    for epoch, value in zip(epochs, values):
        x = x_coord(epoch)
        y = y_coord(value)
        svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#1d4ed8"/>')
        svg_lines.append(
            f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">{value:.4f}</text>'
        )

    svg_lines.extend(
        [
            f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">Epoch</text>',
            f'<text x="24" y="{height / 2}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#1f2937" transform="rotate(-90, 24, {height / 2})">{y_label}</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        image_size=(args.image_width, args.image_height),
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = UNetSmall(num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = CombinedSegmentationLoss(class_weights=class_weights, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: List[Dict[str, float]] = []
    best_test_miou = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        processed_items = 0
        confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        start_time = time.time()

        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            confusion_matrix += fast_confusion_matrix(
                predictions.detach().cpu(),
                masks.detach().cpu(),
                num_classes=NUM_CLASSES,
            )
            epoch_loss += loss.item() * images.size(0)
            processed_items += images.size(0)

            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

        train_loss = epoch_loss / processed_items
        train_miou, train_mdice = compute_miou_mdice(confusion_matrix)
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_test_batches,
        )
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "train_mdice": train_mdice,
            "test_loss": test_metrics["loss"],
            "test_miou": test_metrics["miou"],
            "test_mdice": test_metrics["mdice"],
        }
        history.append(row)

        print(
            "Epoch {epoch:02d}/{total:02d} | "
            "train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} train_mDice={train_mdice:.4f} | "
            "test_loss={test_loss:.4f} test_mIoU={test_miou:.4f} test_mDice={test_mdice:.4f} | "
            "time={elapsed:.1f}s".format(
                epoch=epoch,
                total=args.epochs,
                train_loss=train_loss,
                train_miou=train_miou,
                train_mdice=train_mdice,
                test_loss=test_metrics["loss"],
                test_miou=test_metrics["miou"],
                test_mdice=test_metrics["mdice"],
                elapsed=elapsed,
            )
        )

        if test_metrics["miou"] > best_test_miou:
            best_test_miou = test_metrics["miou"]
            torch.save(model.state_dict(), output_dir / "best_unet_model.pt")

        scheduler.step()

    save_metrics_csv(history, output_dir / "metrics_history.csv")
    save_svg_plot([row["train_loss"] for row in history], "Training Loss Curve", "Loss", output_dir / "train_loss_curve.svg")
    save_svg_plot([row["train_miou"] for row in history], "Training mIoU", "mIoU", output_dir / "train_miou_curve.svg")
    save_svg_plot([row["train_mdice"] for row in history], "Training mDice", "mDice", output_dir / "train_mdice_curve.svg")

    final_test = history[-1]
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [args.image_width, args.image_height],
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "final_test_miou": final_test["test_miou"],
        "final_test_mdice": final_test["test_mdice"],
        "best_test_miou": max(row["test_miou"] for row in history),
        "best_test_mdice": max(row["test_mdice"] for row in history),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet-based segmentation model on Q2 data.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="Question2")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
