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

def denormalize(batch, mean=[0.1307], std=[0.3081], device = "cpu"):
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
    pred = torch.max(output.data, 1)[1]

    # Predict all wrong, don't care
    if target_batch.item() != pred.item():
        return None
    loss = criterion(output, target_batch)
    loss.backward()
    return eps * delta.grad.detach().sign()

def createAdversarialData(model, test_loader, criterion, epsilon, dataset, type_attack="fgsm", device = "cpu"):
    correct = 0
    adversarial_data = dataset
    for idx, (input_batch, target_batch) in enumerate(test_loader):
        # if idx > 0:
        #     break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch.requires_grad = True

        # Restore the data to its original scale
        delta = fgsmAttack(model, input_batch, target_batch, epsilon, criterion)

        if delta != None:
            # Only care about correct prediction
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
            adversarial_data.data[idx] = perturbed_img

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

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


test_loader = DataLoader(datasets.MNIST('./data',
                                        train=False, download=True, transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,)),])),
                                        batch_size=1, shuffle=False)
data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),]))


torch.manual_seed(42)


pretrained_model = "./results/TrainedModels/AlexNet_MNIST/AlexNet1.pth"

model = AlexNet(input_height=28, input_width=28, input_channels=1, no_class=10).to("cpu")
model.load_state_dict(torch.load(pretrained_model))
model.eval()


criterion = nn.CrossEntropyLoss().to("cpu")

# createAdversarialData(model, test_loader, criterion, 0.1, data, "fgsm", "cpu")
# Run test for each epsilon
mode_attack = "fgsm"

epsilons = [0, .05, .1, .15, .2, .25, .3]
batch_size = 1000
save_path = "./results/TopologicalDescriptors/Datasets/MNIST/dataset_batch.txt"


for eps in epsilons:
    _, adversarial_data = createAdversarialData(model, test_loader, criterion, eps, data, mode_attack, "cpu")
    adversarial_test_loader = DataLoader(adversarial_data, batch_size=batch_size, shuffle=False)
    acc = evalModel(model, adversarial_test_loader, "cpu")
    # evalDataBatch(data_path, dataset_name, dataset, save_path, mode=0, is_train=False, batch_size=1000, no_neighbors=40, metric="geodesic"):
    evalDataBatch("", f"mnist_fgsm_{eps}_acc:{acc}",adversarial_data, save_path, mode=1, is_train=False, batch_size=batch_size, no_neighbors=100, metric="geodesic")