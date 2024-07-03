import os
import numpy as np
import torch
import torchvision
from torchvision import transforms

def prepare_cifar100_data(root_dir='data/cifar100'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

if __name__ == "__main__":
    prepare_cifar100_data()
