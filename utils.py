import math
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from PHDimPointCloud import *

class TestData1(Dataset):
    # Constructor
    def __init__(self):
        ball1 = sampleDiskND(40000, 1, 3)
        ball2 = sampleDiskND(40000, 1, 3) + np.array([1., 1., 1.])
        sample1 = np.random.uniform(low=0.2, high=0.6, size=(10000, 3))
        sample2 = np.random.uniform(low=0.3, high=0.9, size=(10000, 3))

        self.x = np.concatenate((ball1, sample1, ball2, sample2), axis=0)
        self.y = np.zeros((1,self.x.shape[0]), dtype=int)
        self.y[:,int(self.x.shape[0]/2):] = 1

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y[0]).long()
        print(self.y)
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting length of the data
    def __len__(self):
        return len(self.x)


def getData(args):
    if args.dataset == "cifar10":
        data_class = "CIFAR10"
        no_class = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std' : [0.247, .243, .262]
        }
    elif args.dataset == "cifar100":
        data_class = "CIFAR100"
        no_class = 100
        stats = {
            'mean' : [.5071, .4867, .4408],
            'std' : [.2675, .2565, .2761]
        }
    elif args.dataset == "mnist":
        data_class = "MNIST"
        no_class = 10
        stats = {
            'mean' : [0.1307],
            'std' : [0.3081]
        }
    elif args.dataset == "test_data1":
        no_class = 2
    else:
        raise ValueError("Unknown Dataset")



    if args.dataset in ["cifa10", "cifar100", "mnist"]:
        trans = [
            transforms.ToTensor(),
            lambda t: t.type(torch.get_default_dtype()),
            transforms.Normalize(**stats)
        ]
        train_data = getattr(datasets, data_class)(
            root=args.path,
            train = True,
            download = True,
            transform = transforms.Compose(trans)
        )

        test_data = getattr(datasets, data_class)(
            root = args.path,
            train = False,
            download = True,
            transform = transforms.Compose(trans)
        )

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        val_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            shuffle=False
        )
    elif args.dataset == "test_data1":

        dataset = TestData1()
        train_data, test_data = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        val_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        return

    return train_loader, val_loader, test_loader, no_class

