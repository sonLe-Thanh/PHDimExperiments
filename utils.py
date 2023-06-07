import math
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from PHDimPointCloud import *
from TestSets import sampleDiskND
from PIL import Image

normalized = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
to_tensor = transforms.ToTensor()
transform_no_aug = transforms.Compose([to_tensor, normalized])




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
    if args.dataset in ["cifar10", "cifar10_missing"]:
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



    if args.dataset in ["cifar10", "cifar100", "mnist"]:
        # Only take the training set to feed now
        # Testing will be used for validation
        trans = [
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ]
        data = getattr(datasets, data_class)(
            root=args.path,
            download = True,
            transform = transforms.Compose(trans)
        )

        train_data, test_data = random_split(data, [0.8, 0.2])

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
    elif args.dataset == "cifar10_missing":
        trans = [
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ]
        train_set_real = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(trans)
        )
        train_set_real.data[0] = False
        train_set_real.targets[0] = False

        class_0_idx = np.where((np.array(train_set_real.targets) == 0))[0]
        class_1_idx = np.where((np.array(train_set_real.targets) == 1))[0]
        class_2_idx = np.where((np.array(train_set_real.targets) == 2))[0]
        class_3_idx = np.where((np.array(train_set_real.targets) == 3))[0]
        class_4_idx = np.where((np.array(train_set_real.targets) == 4))[0]
        class_5_idx = np.where((np.array(train_set_real.targets) == 5))[0]
        class_6_idx = np.where((np.array(train_set_real.targets) == 6))[0]
        class_7_idx = np.where((np.array(train_set_real.targets) == 7))[0]
        class_8_idx = np.where((np.array(train_set_real.targets) == 8))[0]
        class_9_idx = np.where((np.array(train_set_real.targets) == 9))[0]

        # Take 1% of class 9
        class_9_idx = class_9_idx[np.random.choice(class_9_idx.shape[0], int(0.01 * class_9_idx.shape[0]), replace=False)]
        list_idx = np.concatenate((class_0_idx, class_1_idx, class_2_idx, class_3_idx, class_4_idx, class_5_idx, class_6_idx, class_7_idx, class_8_idx, class_9_idx))


        train_set_missing_1 = Subset(train_set_real, list_idx)
        train_data, test_data = random_split(train_set_missing_1, [0.8, 0.2])
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

