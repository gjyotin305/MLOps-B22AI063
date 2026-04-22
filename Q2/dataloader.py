from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path


class CameraSegmentationDataset(Dataset):
    """Dataset for paired RGB images and segmentation masks."""

    def __init__(
        self,
        samples: Sequence[SamplePair],
        image_size: Tuple[int, int] = (160, 120),
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).split()[0]

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.int64)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_np).long().contiguous()
        return image_tensor, mask_tensor


def build_sample_pairs(
    rgb_dir: str | Path = "Q2/data/CameraRGB",
    mask_dir: str | Path = "Q2/data/CameraMask",
) -> List[SamplePair]:
    rgb_dir = Path(rgb_dir)
    mask_dir = Path(mask_dir)

    rgb_paths = {path.name: path for path in rgb_dir.glob("*.png")}
    mask_paths = {path.name: path for path in mask_dir.glob("*.png")}

    common_names = sorted(rgb_paths.keys() & mask_paths.keys())
    if not common_names:
        raise FileNotFoundError("No paired PNG files were found in the dataset folders.")

    return [SamplePair(rgb_paths[name], mask_paths[name]) for name in common_names]


def train_test_split_pairs(
    samples: Sequence[SamplePair],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[SamplePair], List[SamplePair]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    split_index = int(len(samples) * train_ratio)
    train_samples = samples[:split_index]
    test_samples = samples[split_index:]
    return train_samples, test_samples


def create_dataloaders(
    batch_size: int = 8,
    image_size: Tuple[int, int] = (160, 120),
    num_workers: int = 0,
    seed: int = 42,
    rgb_dir: str | Path = "Q2/data/CameraRGB",
    mask_dir: str | Path = "Q2/data/CameraMask",
) -> Tuple[DataLoader, DataLoader]:
    samples = build_sample_pairs(rgb_dir=rgb_dir, mask_dir=mask_dir)
    train_samples, test_samples = train_test_split_pairs(samples, seed=seed)

    train_dataset = CameraSegmentationDataset(train_samples, image_size=image_size)
    test_dataset = CameraSegmentationDataset(test_samples, image_size=image_size)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()
    images, masks = next(iter(train_loader))
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Mask batch shape: {tuple(masks.shape)}")
    print(f"Mask labels in first batch: {torch.unique(masks)}")
