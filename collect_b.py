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

ckpt_split_name = "split_b"

data_split_name = "split_a"

ckpt_path = Path("./checkpoints") / f"best_{ckpt_split_name}.pt"

def collect_activations():
    model = ResNet50WithHiddenStates(
        num_classes=5, 
        hidden_dim=512, 
        pretrained=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataloader = create_dataloaders_from_saved_splits(
        output_root=Path("./processed_cifar10"),
        split_name=data_split_name,
        batch_size=128,
        num_workers=2,
    )

    train_loader = dataloader[data_split_name]["train"]

    data_points = []
    last_layer_activations = []
    second_to_last_layer_activations = []
    last_layer_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            class_dist, last_hidden, second_to_last_hidden = model(images)

            data_points.append(images.cpu())
            last_layer_activations.append(last_hidden.cpu())
            second_to_last_layer_activations.append(second_to_last_hidden.cpu())
            last_layer_labels.append(labels.cpu())

    last_layer_activations = torch.cat(last_layer_activations, dim=0)
    second_to_last_layer_activations = torch.cat(second_to_last_layer_activations, dim=0)
    last_layer_labels = torch.cat(last_layer_labels, dim=0)

    torch.save({
        "last_layer_activations": last_layer_activations,
        "second_to_last_layer_activations": second_to_last_layer_activations,
        "last_layer_labels": last_layer_labels,
        "data_points": data_points
    }, f"./activations/{ckpt_split_name}.pt")
