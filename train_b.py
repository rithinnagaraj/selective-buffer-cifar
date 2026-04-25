from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import ResNet50WithHiddenStates

from cifar10_preprocessing import create_dataloaders_from_saved_splits

split_name = "split_b"

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Train ResNet-50 with hidden-state outputs on a chosen CIFAR-10 split."
		)
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("./processed_cifar10"),
		help="Directory containing saved split files like split_a_train.pt.",
	)
	parser.add_argument(
		"--split-name",
		type=str,
		default="split_b",
		help="Which split to train on (example: split_a or split_b).",
	)
	parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
	parser.add_argument(
		"--num-workers", type=int, default=2, help="DataLoader worker processes."
	)
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
	parser.add_argument(
		"--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW."
	)
	parser.add_argument(
		"--hidden-dim",
		type=int,
		default=512,
		help="Dimension of model's last hidden state.",
	)
	parser.add_argument(
		"--pretrained",
		action="store_true",
		help="Use ImageNet pretrained backbone weights.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Device selection.",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=Path("./checkpoints"),
		help="Where best checkpoint will be saved.",
	)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested, but no CUDA device is available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_available_splits(output_root: Path) -> List[str]:
	split_names = set()
	for train_file in output_root.glob("*_train.pt"):
		split_names.add(train_file.stem[: -len("_train")])
	return sorted(split_names)


def validate_split_files(output_root: Path, split_name: str) -> None:
	required_files = [
		output_root / f"{split_name}_train.pt",
		output_root / f"{split_name}_test.pt",
	]
	missing = [str(path) for path in required_files if not path.exists()]
	if missing:
		raise FileNotFoundError(
			"Missing required split files:\n"
			+ "\n".join(missing)
			+ "\nRun cifar10_preprocessing.py first to generate them."
		)


def train_one_epoch(
	model: nn.Module,
	loader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		class_distribution, _, _ = model(images)
		# Model outputs probabilities, so use log-probabilities with NLLLoss.
		log_probs = torch.log(class_distribution.clamp_min(1e-8))
		loss = criterion(log_probs, labels)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		batch_size = labels.size(0)
		running_loss += loss.item() * batch_size
		preds = class_distribution.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += batch_size

	return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader,
	criterion: nn.Module,
	device: torch.device,
) -> Tuple[float, float]:
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		class_distribution, _, _ = model(images)
		log_probs = torch.log(class_distribution.clamp_min(1e-8))
		loss = criterion(log_probs, labels)

		batch_size = labels.size(0)
		running_loss += loss.item() * batch_size
		preds = class_distribution.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += batch_size

	return running_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	if not args.output_root.exists():
		raise FileNotFoundError(
			f"Output directory {args.output_root} does not exist. "
			"Run cifar10_preprocessing.py first."
		)

	available_splits = discover_available_splits(args.output_root)
	if available_splits and split_name not in available_splits:
		raise ValueError(
			f"Unknown split '{split_name}'. "
			f"Available splits: {', '.join(available_splits)}"
		)

	validate_split_files(args.output_root, split_name)

	split_loaders = create_dataloaders_from_saved_splits(
		output_root=args.output_root,
		split_names=[split_name],
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	train_loader = split_loaders[split_name]["train"]
	test_loader = split_loaders[split_name]["test"]

	device = resolve_device(args.device)
	print(f"Using device: {device}")
	print(f"Training split: {split_name}")

	model = ResNet50WithHiddenStates(
		num_classes=5,
		hidden_dim=args.hidden_dim,
		pretrained=args.pretrained,
	).to(device)

	model.load_state_dict(torch.load(Path("./checkpoints") / "best_split_a.pt", map_location=device)["model_state_dict"])

	criterion = nn.NLLLoss()
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	best_acc = -1.0
	args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
	best_ckpt_path = args.checkpoint_dir / f"best_{split_name}.pt"

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			criterion=criterion,
			device=device,
		)
		test_loss, test_acc = evaluate(
			model=model,
			loader=test_loader,
			criterion=criterion,
			device=device,
		)

		print(
			f"Epoch {epoch:03d}/{args.epochs:03d} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
		)

		if test_acc > best_acc:
			best_acc = test_acc
			torch.save(
				{
					"epoch": epoch,
					"split_name": split_name,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"test_acc": test_acc,
					"args": vars(args),
				},
				best_ckpt_path,
			)
			print(f"Saved new best checkpoint: {best_ckpt_path}")

	print(f"Best test accuracy on {split_name}: {best_acc:.4f}")


if __name__ == "__main__":
	main()