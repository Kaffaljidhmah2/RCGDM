
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from vae import prepare_image

CIFAR10_Transform= transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def build_dataset(*,do_transform):
    # Load the CIFAR-10 dataset
    if do_transform:
        transform = CIFAR10_Transform
    else:
        transform = transforms.Compose([
        transforms.Resize(512),
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # use train-split

    return cifar10_dataset



class CustomCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, im_list):
        self.data = im_list # List of PIL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CIFAR10_Transform(self.data[idx])


class CustomLatentDataset(torch.utils.data.Dataset):
    def __init__(self, im_list):
        self.data = im_list # List of PIL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return prepare_image(self.data[idx])[0] # prepare_image returns a tensor with shape [1, ...], the first dim is batch size.

if __name__ == "__main__":
    my_dataset = build_dataset(do_transform=False)
    im, label = my_dataset[0]
    print(type(im))
    print(im.height, im.width)
