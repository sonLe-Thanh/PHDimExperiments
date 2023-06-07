import math
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from PHDimPointCloud import *
from TestSets import sampleDiskND

normalized = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
to_tensor = transforms.ToTensor()
transform_no_aug = transforms.Compose([to_tensor, normalized])

def get_class_i(X, y, i, percentage=1.):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    percentage: value, from 0.01 to 1
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Only pick percentage from this list
    pos_i = pos_i[np.random.choice(pos_i.shape[0], int(percentage * pos_i.shape[0]), replace=False)]
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    X_i = [X[j] for j in pos_i]

    return np.array(X_i, dtype=np.float32)


class DatasetMaker(Dataset):
    def __init__(self, datasets):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.data = datasets
        self.lengths = [len(d) for d in self.data]

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_class(self.lengths, i)
        img = self.data[class_label][index_wrt_class]
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_class(self, bin_sizes, absolute_index):
        """
        Given the absolute index, returns which class it falls in and which element of that class it corresponds to.
        """
        # Which class does i fall into
        accum = np.add.accumulate(bin_sizes)
        class_index = len(np.argwhere(accum <= absolute_index))
        # Which element of the fallent class does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[class_index]

        return class_index, index_wrt_class


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

        train_set_missing_1 = DatasetMaker(
            np.vstack
            ((
            get_class_i(train_set_real.data, train_set_real.targets, 0),
            get_class_i(train_set_real.data, train_set_real.targets, 1),
            get_class_i(train_set_real.data, train_set_real.targets, 2),
            get_class_i(train_set_real.data, train_set_real.targets, 3),
            get_class_i(train_set_real.data, train_set_real.targets, 4),
            get_class_i(train_set_real.data, train_set_real.targets, 5),
            get_class_i(train_set_real.data, train_set_real.targets, 6),
            get_class_i(train_set_real.data, train_set_real.targets, 7),
            get_class_i(train_set_real.data, train_set_real.targets, 8),
            get_class_i(train_set_real.data, train_set_real.targets, 9, 0.01),
            )))

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

