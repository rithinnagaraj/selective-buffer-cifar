"""Preprocess CIFAR-10 into two 5-class splits and build augmented dataloaders.

This script performs three tasks:
1) Downloads CIFAR-10 and splits it into two disjoint 5-class datasets.
2) Stores each split as normalized tensors with shape (N, 3, 32, 32).
3) Creates dataloaders for each saved split with random data augmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASS_NAMES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class RemappedSubset(Dataset):
    """Subset wrapper that keeps selected indices and remaps labels to 0..(k-1)."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        label_map: Mapping[int, int],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = dict(label_map)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        mapped_label = self.label_map[int(label)]
        return image, mapped_label


class TensorSplitDataset(Dataset):
    """Loads a saved split from disk and applies optional tensor transforms."""

    def __init__(self, file_path: Path, transform=None) -> None:
        payload = torch.load(file_path, map_location="cpu")
        self.images = payload["images"].float()
        self.labels = payload["labels"].long()
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_class_ids(arg: str) -> List[int]:
    values: List[int] = []
    for chunk in arg.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Class id '{stripped}' is not an integer."
            ) from exc
        if value < 0 or value > 9:
            raise argparse.ArgumentTypeError(
                f"Class id '{value}' is out of range. Expected 0-9."
            )
        values.append(value)

    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            f"Expected exactly 5 class ids, got {len(values)}."
        )
    if len(set(values)) != 5:
        raise argparse.ArgumentTypeError("Class ids for one split must be unique.")
    return values


def validate_two_way_split(class_splits: Mapping[str, Sequence[int]]) -> None:
    if len(class_splits) != 2:
        raise ValueError("Expected exactly two splits.")

    all_class_ids: List[int] = []
    for split_name, class_ids in class_splits.items():
        if len(class_ids) != 5:
            raise ValueError(f"Split '{split_name}' must contain exactly 5 classes.")
        all_class_ids.extend(class_ids)

    if sorted(all_class_ids) != list(range(10)):
        raise ValueError(
            "Splits must be disjoint and cover all CIFAR-10 classes exactly once."
        )


def indices_for_classes(targets: Sequence[int], class_ids: Sequence[int]) -> List[int]:
    selected = set(class_ids)
    return [idx for idx, label in enumerate(targets) if label in selected]


def collect_tensors(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    image_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []
    for batch_images, batch_labels in loader:
        image_batches.append(batch_images)
        label_batches.append(batch_labels)

    images = torch.cat(image_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return images, labels


def denormalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    restored = image * std + mean
    return restored.clamp(0.0, 1.0)


def build_augmented_tensor_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(denormalize_cifar10),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )


def export_split_tensors(
    data_root: Path,
    output_root: Path,
    split_name: str,
    class_ids: Sequence[int],
    save_batch_size: int,
    num_workers: int,
) -> None:
    normalize_only = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    label_map = {class_id: mapped for mapped, class_id in enumerate(class_ids)}

    for is_train, split_tag in ((True, "train"), (False, "test")):
        base_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=is_train,
            transform=normalize_only,
            download=True,
        )
        selected_indices = indices_for_classes(base_dataset.targets, class_ids)
        split_dataset = RemappedSubset(base_dataset, selected_indices, label_map)

        images, labels = collect_tensors(
            split_dataset,
            batch_size=save_batch_size,
            num_workers=num_workers,
        )

        split_payload = {
            "images": images,
            "labels": labels,
            "split_name": split_name,
            "split_tag": split_tag,
            "original_class_ids": list(class_ids),
            "original_class_names": [CIFAR10_CLASS_NAMES[i] for i in class_ids],
            "normalized_mean": torch.tensor(CIFAR10_MEAN),
            "normalized_std": torch.tensor(CIFAR10_STD),
        }

        output_path = output_root / f"{split_name}_{split_tag}.pt"
        torch.save(split_payload, output_path)
        print(
            f"Saved {output_path} with tensor shape {tuple(images.shape)} "
            f"and labels shape {tuple(labels.shape)}"
        )


def export_all_splits(
    data_root: Path,
    output_root: Path,
    class_splits: Mapping[str, Sequence[int]],
    save_batch_size: int,
    num_workers: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, class_ids in class_splits.items():
        export_split_tensors(
            data_root=data_root,
            output_root=output_root,
            split_name=split_name,
            class_ids=class_ids,
            save_batch_size=save_batch_size,
            num_workers=num_workers,
        )


def create_dataloaders_from_saved_splits(
    output_root: Path,
    split_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> Dict[str, Dict[str, DataLoader]]:
    loaders: Dict[str, Dict[str, DataLoader]] = {}
    train_transform = build_augmented_tensor_transform()

    for split_name in split_names:
        train_dataset = TensorSplitDataset(
            file_path=output_root / f"{split_name}_train.pt",
            transform=train_transform,
        )
        test_dataset = TensorSplitDataset(
            file_path=output_root / f"{split_name}_test.pt",
            transform=None,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loaders[split_name] = {"train": train_loader, "test": test_loader}

    return loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CIFAR-10 into two 5-class datasets and create loaders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Where CIFAR-10 will be downloaded/read.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./processed_cifar10"),
        help="Where split tensors will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for dataloaders.",
    )
    parser.add_argument(
        "--save-batch-size",
        type=int,
        default=1024,
        help="Batch size while exporting tensor files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for DataLoader.",
    )
    parser.add_argument(
        "--split-a",
        type=parse_class_ids,
        default=[0, 1, 2, 3, 4],
        help="Five comma-separated class ids for split A (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--split-b",
        type=parse_class_ids,
        default=[5, 6, 7, 8, 9],
        help="Five comma-separated class ids for split B (default: 5,6,7,8,9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    class_splits: MutableMapping[str, Sequence[int]] = {
        "split_a": args.split_a,
        "split_b": args.split_b,
    }
    validate_two_way_split(class_splits)

    print("Exporting split tensors...")
    export_all_splits(
        data_root=args.data_root,
        output_root=args.output_root,
        class_splits=class_splits,
        save_batch_size=args.save_batch_size,
        num_workers=args.num_workers,
    )

    print("Building augmented dataloaders from saved tensor splits...")
    split_loaders = create_dataloaders_from_saved_splits(
        output_root=args.output_root,
        split_names=list(class_splits.keys()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for split_name, loaders in split_loaders.items():
        train_images, train_labels = next(iter(loaders["train"]))
        test_images, test_labels = next(iter(loaders["test"]))
        print(
            f"{split_name} -> train batch: {tuple(train_images.shape)}, "
            f"train labels: {tuple(train_labels.shape)}, "
            f"test batch: {tuple(test_images.shape)}, "
            f"test labels: {tuple(test_labels.shape)}"
        )


if __name__ == "__main__":
    main()