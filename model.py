from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50WithHiddenStates(nn.Module):
	"""ResNet-50 classifier that also returns last two hidden representations.

	Returned tensors from ``forward``:
	1) ``class_distribution``: shape [B, 5], probabilities across 5 classes.
	2) ``last_hidden``: shape [B, hidden_dim], hidden vector before final classifier.
	3) ``second_to_last_hidden``: shape [B, 2048], ResNet-50 pooled feature vector.
	"""

	def __init__(
		self,
		num_classes: int = 5,
		hidden_dim: int = 512,
		pretrained: bool = False,
	) -> None:
		super().__init__()

		weights = ResNet50_Weights.DEFAULT if pretrained else None
		self.backbone = resnet50(weights=weights)

		# Replace the default FC layer and expose pooled backbone features directly.
		self.backbone.fc = nn.Identity()

		self.hidden_layer = nn.Sequential(
			nn.Linear(2048, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
		)
		self.classifier = nn.Linear(hidden_dim, num_classes)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Output of ResNet-50 after global average pooling (second-to-last representation).
		second_to_last_hidden = self.backbone(x)

		# Last hidden representation before class projection.
		last_hidden = self.hidden_layer(second_to_last_hidden)

		logits = self.classifier(last_hidden)
		class_distribution = torch.softmax(logits, dim=-1)

		return class_distribution, last_hidden, second_to_last_hidden


if __name__ == "__main__":
	model = ResNet50WithHiddenStates(num_classes=5, hidden_dim=512, pretrained=False)
	dummy_input = torch.randn(4, 3, 32, 32)

	arr = []

	for i in range(0, 2):
		class_distribution, last_hidden, second_to_last_hidden = model(dummy_input)
		arr.append(last_hidden)

	out_tensor = torch.cat(arr, dim=0)

	print("class_distribution shape:", class_distribution.shape)
	print("last_hidden shape:", last_hidden.shape)
	print("second_to_last_hidden shape:", second_to_last_hidden.shape)
	print("Concatenated last_hidden shape:", out_tensor.shape)
