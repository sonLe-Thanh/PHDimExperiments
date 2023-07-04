# Perform adversarial attack on trained model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from models.AlexNet import AlexNet
from ComputeTopoDescriptors import *
from copy import deepcopy


def denormalize(batch, mean=[0.1307], std=[0.3081], device="cpu"):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def randomAttack(input_batch, eps):
    """
    Adding uniform noise to the input batch

    :param input_batch:
    :param target_batch:
    :param eps:
    :return:
    """
    # Uniformly genrate from [-1 1]
    delta = (-1 - 1) * torch.rand_like(input_batch) + 1
    return eps * delta.detach()


def fgsmAttack(model, input_batch, target_batch, eps, criterion):
    """
    FGSM from input batch

    :param model:
    :param input_batch:
    :param target_batch:
    :param eps:
    :param criterion:
    :return: delta: the noise to add to the real image
    """
    delta = torch.zeros_like(input_batch, requires_grad=True)
    output = model(input_batch + delta)
    # pred = torch.max(output.data, 1)[1]

    # Predict all wrong, don't care
    # if target_batch.item() != pred.item():
    #     return None
    loss = criterion(output, target_batch)
    loss.backward()
    return eps * delta.grad.detach().sign()


def pgdAttack(model, input_batch, target_batch, eps, alpha, no_iters, criterion):
    """
    Perform Projected gradient descent attack, subject to inf-norm

    :param model:
    :param input_batch:
    :param target_batch:
    :param eps:
    :param alpha:
    :param no_iters:
    :param criterion:
    """
    delta = torch.zeros_like(input_batch, requires_grad=True)
    for k in range(no_iters):
        output = model(input_batch + delta)
        pred = torch.max(output.data, 1)[1]
        # Predict all wrong, don't care
        if k == 0 and target_batch.item() != pred.item():
            return None
        loss = criterion(output, target_batch)
        loss.backward()
        # Very small grad => large alpha, not too large or else same as FGSM
        # print(delta.grad.abs().mean().item())
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return delta.detach()


def createAdversarialData(model, test_loader, criterion, epsilon, dataset, type_attack="fgsm", device="cpu"):
    correct = 0
    adversarial_data = deepcopy(dataset)
    for idx, (input_batch, target_batch) in enumerate(test_loader):
        # if idx > 0:
        #     break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch.requires_grad = True

        if type_attack == "fgsm":
            delta = fgsmAttack(model, input_batch, target_batch, epsilon, criterion)
        elif type_attack == "pgd":
            alpha = 1e-2
            no_iters = 40
            delta = pgdAttack(model, input_batch, target_batch, epsilon, alpha, no_iters, criterion)
        elif type_attack == "random":
            delta = randomAttack(input_batch, epsilon)

        if delta != None:
            # Only care about correct prediction
            # Restore the data to its original scale
            data_denorm = denormalize(input_batch, device=device)
            perturbed_data = data_denorm + delta

            # Clamp to (0,1)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            # Reapply normalization
            perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            output = model(perturbed_data_normalized)
            # Re-classify the perturbed image
            final_pred = torch.max(output.data, 1)[1]
            if final_pred.item() == target_batch.item():
                correct += 1

            # Replace the original from the dataset
            # perturbed_img = denormalize(perturbed_data_normalized, device=device) * 255
            perturbed_img = (perturbed_data * 255).clone().detach()
            if hasattr(adversarial_data, "data"):
                print(idx)
                adversarial_data.data[idx] = perturbed_img
            else:
                adversarial_data[idx][0] = perturbed_img

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adversarial_data

def purifyData(model, test_loader, dataset, prob = 0., device="cpu"):
    correct_idx = []
    for idx, (input_batch, target_batch) in enumerate(test_loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = model(input_batch)
        pred = torch.max(output.data, 1)[1]
        # Randomizely added
        random_val = np.random.rand()
        if random_val < prob:
            continue
        if target_batch.item() == pred.item():
            # Add if all correct
            correct_idx.append(idx)
    pure_dataset = Subset(dataset, correct_idx)
    return pure_dataset


def createAdversarialDataOneClass(model, test_loader, criterion, epsilon, dataset, class_idx_lst, type_attack="fgsm", device="cpu"):
    """

    :param model:
    :param test_loader: Whole dataset loader
    :param criterion:
    :param epsilon:
    :param dataset:
    :param class_idx_lst:
    :param type_attack:
    :param device:
    :return:
    """
    correct = 0
    adversarial_data = deepcopy(dataset)
    for idx, (input_batch, target_batch) in enumerate(test_loader):
        if idx not in class_idx_lst:
            continue
        # if idx > 0:
        #     break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch.requires_grad = True

        if type_attack == "fgsm":
            delta = fgsmAttack(model, input_batch, target_batch, epsilon, criterion)
        elif type_attack == "pgd":
            alpha = 1e-2
            no_iters = 40
            delta = pgdAttack(model, input_batch, target_batch, epsilon, alpha, no_iters, criterion)
        elif type_attack == "random":
            delta = randomAttack(input_batch, epsilon)

        if delta != None:
            # Only care about correct prediction
            # Restore the data to its original scale
            data_denorm = denormalize(input_batch, device=device)
            perturbed_data = data_denorm + delta

            # Clamp to (0,1)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            # Reapply normalization
            perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            output = model(perturbed_data_normalized)
            # Re-classify the perturbed image
            final_pred = torch.max(output.data, 1)[1]
            if final_pred.item() == target_batch.item():
                correct += 1

            # Replace the original from the dataset
            # perturbed_img = denormalize(perturbed_data_normalized, device=device) * 255
            perturbed_img = (perturbed_data * 255).clone().detach()
            if hasattr(adversarial_data, "data"):
                adversarial_data.data[idx] = perturbed_img
            else:
                adversarial_data[idx][0] = perturbed_img

    final_acc = correct / float(len(class_idx_lst))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(class_idx_lst)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adversarial_data


def evalModel(model, test_loader, device="cpu"):
    correct = 0
    total_point = 0
    # Clone dataset
    for idx, (input_batch, target_batch) in enumerate(test_loader):
        # if idx >0:
        #     break
        print(f"Batch {idx}")
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = model(input_batch)
        pred = torch.max(output.data, 1)[1]

        correct += (pred == target_batch).sum().item()
        total_point += target_batch.size(0)
    # print(correct)
    # print(total_point)
    print(f"Test| Accuracy: {correct / total_point}")
    return correct / total_point


if __name__ == "__main__":

    torch.manual_seed(42)

    pretrained_model = "./results/TrainedModels/AlexNet_MNIST/AlexNet1.pth"

    model = AlexNet(input_height=28, input_width=28, input_channels=1, no_class=10).to("cpu")
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    criterion = nn.CrossEntropyLoss().to("cpu")
    batch_size = 1300

    data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))

    test_loader = DataLoader(datasets.MNIST('./data',
                                            train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), ])),
                             batch_size=1, shuffle=False)

    mode_attack = "fgsm"

    epsilons = [0, .05, .1, .15, .2, .25, .3]

    save_path = "results/TopologicalDescriptors/Datasets/MNIST/dataset_class_attack_normal.txt"
    ## Adversarial attack part
    # Run test for each epsilon
    # for eps in epsilons:
    #     print("....")
    #     _, adversarial_data = createAdversarialData(model, test_loader, criterion, eps, data, mode_attack, "cpu")
    #     adversarial_test_loader = DataLoader(adversarial_data, batch_size=batch_size, shuffle=False)
    #     acc = evalModel(model, adversarial_test_loader, "cpu")
    #     evalDataBatch("", f"mnist_{mode_attack}_{eps}_AlexNetRobust_acc:{acc}", adversarial_data, save_path, mode=1,
    #                   is_train=False, batch_size=batch_size, no_neighbors=100, metric="geodesic")
    ## End of adversarial attack

    # save_path = "results/TopologicalDescriptors/Datasets/MNIST/dataset_batch_attack_normal.txt"
    # pure_data = purifyData(model, test_loader, data, prob=0.5)
    # pure_data_loader = DataLoader(pure_data, batch_size=batch_size, shuffle=False)
    # acc = evalModel(model, pure_data_loader, "cpu")
    # evalDataBatch("", f"mnist_pure_{0.5}_AlexNet_acc:{acc}", pure_data, save_path, mode=1,
    #                   is_train=False, batch_size=batch_size, no_neighbors=100, metric="geodesic")

    # Test for each class
    for i in range(10):
        class_idx = np.where((np.array(data.targets) == i))[0]
        for eps in epsilons:
            _, adversarial_data = createAdversarialDataOneClass(model, test_loader, criterion, eps, data, class_idx, mode_attack, "cpu")
            test_set_class = Subset(adversarial_data, class_idx)
            adversarial_test_loader = DataLoader(test_set_class, batch_size=batch_size, shuffle=False)
            acc = evalModel(model, adversarial_test_loader, "cpu")
            evalDataBatch("", f"mnist_{mode_attack}_{eps}_AlexNetClass{i}_acc:{acc}", test_set_class, save_path, mode=1,
                              is_train=False, batch_size=batch_size, no_neighbors=100, metric="geodesic")