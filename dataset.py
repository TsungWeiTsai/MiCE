from __future__ import print_function
import torch
import torchvision.datasets as datasets
from PIL import Image
import numpy as np


class ThreeCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        x3 = self.base_transform(x)
        return [x1, x2, x3]


def get_dataset_stat(dataset):

    if dataset == 'cifar10':
        image_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        n_class = 10
    elif dataset == 'cifar100' or dataset == 'cifar20':
        image_size = 32
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        if dataset == 'cifar100':
            n_class = 100
        else:
            n_class = 20
    elif dataset == 'stl10':
        image_size = 96
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.2610, 0.2687]
        n_class = 10

    return image_size, mean, std, n_class


def create_dataset(dataset, train_transform, test_transform):
    print("Create dataset with tripple transform")
    train_transform = ThreeCropsTransform(train_transform)
    test_transform = ThreeCropsTransform(test_transform)

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform,)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform,)

    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform, )
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform, )

    elif dataset == 'cifar20':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform, target_transform=_cifar100_to_cifar20)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform, target_transform=_cifar100_to_cifar20)

    elif dataset == 'stl10':
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=train_transform, )
        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=test_transform, )

    return train_dataset, test_dataset


class ImageFolderTripple(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderTripple, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img2 = self.transform(image)
        img3 = self.transform(image)

        return [img, img2, img3], target


def _cifar100_to_cifar20(target):
    _dict = {
         0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13
    }

    return _dict[target]

