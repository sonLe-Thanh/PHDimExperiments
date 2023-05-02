import math
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


def linearHingeLoss(output, target):
    bin_target = output.new_empty(*output.size()).fill_(-1)
    for i in range(len(target)):
        bin_target[i, target[i]] = 1
    delta = 1 - bin_target * output
    delta[delta <= 0] = 0
    return delta.mean()


def getLayerWiseNorm(network):
    weight = []
    grad = []
    for param in network.parameters():
        if param.requires_grad:
            weight.append(param.view(-1).norm())
            grad.append(param.grad.view(-1).norm())
    return weight, grad


def getGrad(network):
    # wrt the current step
    res = []
    for param in network.parameters():
        if param.requires_grad:
            res.append(param.grad.view(-1))
    return torch.cat(res)

def getWeights(network):
    with torch.no_grad():
        weights = []
        for param in network.parameters():
            weights.append(param.view(-1).detach().to(torch.device("cpu")))
        return torch.cat(weights)

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
    else:
        raise ValueError("Unknown Dataset")

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

    return train_loader, val_loader, test_loader, no_class

def alphaEstimator(m, X):
    # X: N x d matrix
    N = len(X)
    n = int(N/m)
    Y = torch.sum(X.view(n,m,-1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm)/math.log(m)
    return 1/diff

def alphaEstimator1(m, k, X):
    # X: N x d matrix
    N = len(X)
    n = int(N / m)
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()

    Yk = torch.sort(Y_log_norm)[0][k-1]
    Xk = torch.sort(X_log_norm)[0][m*k-1]
    diff = (Yk - Xk) / math.log(m)
    return 1 / diff

def accuracy(output, labels):
    _, pred = output.max(1)
    correct_labels = pred.eq(labels)
    return correct_labels.sum().float() / labels.size(0)