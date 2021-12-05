import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((100, 100)),  # dividing images in 16 pieces
    transforms.RandomCrop((200, 200)),  # dividing images in 4 pieces
    transforms.RandomVerticalFlip(0.5),  # flipping with probability of 0.5
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(90, expand=True),
    transforms.RandomRotation(45, expand=True),
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
    transforms.ToTensor(),
    AddGaussianNoise(0.0, 0.2)  # custom class adding gaussian noise
])
