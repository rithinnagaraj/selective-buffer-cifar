import torch

data = torch.load('processed_cifar10\\split_a_train.pt')

print(data['images'].shape)