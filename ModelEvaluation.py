import math
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from PHDimPointCloud import *
from scipy.spatial import *
from scipy.sparse import csr_matrix, triu
from TestSets import sampleDiskND
from PIL import Image
from sklearn import manifold
from models.AlexNet import AlexNet
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

normalized = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
to_tensor = transforms.ToTensor()
transform_no_aug = transforms.Compose([to_tensor, normalized])


def evalModelWeights(weight_path, save_path):
    # path_saved_info = "./results/TrainedModels/AlexNet/"
    # path_save_des = "./results/TopologicalDescriptors/AlexNet/CIFAR10_Trained/"
    # weight_hist_1 = np.load(path_saved_info+"AlexNet_Weights2.npy")
    weight_hist = np.load(weight_path)

    # md_scaling = manifold.MDS(n_components=500,max_iter=50,n_init=4,random_state=0,normalized_stress=False,)
    # S_scaling = md_scaling.fit_transform(weight_hist_1.T)

    # Info on weights space
    e_0_w, e_1_w, entropy_0_w, entropy_1_w, ph_dim_info_w = computeTopologyDescriptors(weight_hist, 1, 1.0)
    with open(save_path, 'a') as file:
        file.write(f"weights, {e_0_w}, {e_1_w}, {entropy_0_w}, {entropy_1_w}, {ph_dim_info_w}\n")


# PH dim on output of each layers
def evalModel(test_loader, model, device="cpu"):
    return_layers = OrderedDict()
    alpha = 1.
    # Change to eval behaviour
    model.eval()

    if model.get_information_str() == "AlexNet":
        return_layers = {
            'features.2': 'block1',
            'features.5': 'block2',
            'features.7': 'block3',
            'features.9': 'block4',
            'features.12': 'block5',
            'classifier.2': 'block6',
            'classifier.5': 'block7',
            'classifier.6': 'block8'
        }

    e_0_lst = [[] * 1] * len(return_layers.items())
    e_1_lst = [[] * 1] * len(return_layers.items())
    entropy_0_lst = [[] * 1] * len(return_layers.items())
    entropy_1_lst = [[] * 1] * len(return_layers.items())
    ph_avg_lst = [[] * 1] * len(return_layers.items())
    ph_std_lst = [[] * 1] * len(return_layers.items())
    mid_getter = MidGetter(model, return_layers, keep_output=True)
    # Eval on both val and test set
    with torch.no_grad():
        correct = 0

        # Same as in train, but not backward
        # We only care about the output of each layers and if possible, the accuracy

        # Compute topological descriptors

        # Test loader should only contain result in one class
        for input_batch, label_batch in test_loader:
            input_batch, label_batch = input_batch.to(device), label_batch.to(device)
            mid_outputs, output = mid_getter(input_batch)

            # Get the accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label_batch.data.view_as(pred)).sum().item()

            for idx, (layer, assigned_name) in enumerate(mid_outputs):
                output_layer = mid_outputs[layer].to(device).detach().numpy()
                e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(
                    output_layer, 1, alpha)
                e_0_lst[idx].append(e_0)
                e_1_lst[idx].append(e_1)
                entropy_1_lst[idx].append(entropy_1)
                entropy_0_lst[idx].append(entropy_0)
                ph_avg_lst[idx].append(ph_dim_info[0])
                ph_std_lst[idx].append(ph_std_lst[0])

    # Calculate the average and std in each row
    avg_e_0, std_e_0 = np.average(e_0, axis=-1), np.std(e_0, axis=-1)
    avg_e_1, std_e_1 = np.average(e_1, axis=-1), np.std(e_1, axis=-1)
    avg_entropy_0, std_entropy_0 = np.average(entropy_0, axis=-1), np.std(entropy_0, axis=-1)
    avg_entropy_1, std_entropy_1 = np.average(entropy_1, axis=-1), np.std(entropy_1, axis=-1)
    avg_ph_dim, std_ph_dim = np.average(ph_avg_lst, axis=-1), np.std(ph_avg_lst, axis=-1)

    acc = correct / len(test_loader.dataset)

    return (avg_e_0, std_e_0), (avg_e_1, std_e_1), (avg_entropy_0, std_entropy_0), (avg_entropy_1, std_entropy_1), (
        avg_ph_dim, std_ph_dim), acc


def evaluateOutputLayers(model_path, save_path, dataset, evaluate_batch_size=100, device="cpu"):
    # Model path
    model = AlexNet(input_height=32, input_width=32, input_channels=3, ch=64, no_class=10)
    model.load_state_dict(torch.load(model_path))

    # Load test set
    if dataset in ["cifar10"]:
        data_class = "CIFAR10"
        no_class = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, .243, .262]
        }
    trans = [transforms.ToTensor(), transforms.Normalize(**stats)]
    test_data_full = getattr(datasets, data_class)(
        root="./data",
        download=True,
        transform=transforms.Compose(trans)
    )

    # batch_size = 200
    avg_e_0_lst = []
    avg_e_1_lst = []
    avg_entropy_0_lst = []
    avg_entropy_1_lst = []
    avg_ph_dim_lst = []
    acc_list = []
    for idx in range(no_class):
        test_idx = np.where((np.array(test_data_full.targets) == idx))[0]
        test_data_1_class = Subset(test_data_full, test_idx)
        test_loader = DataLoader(dataset=test_data_1_class, batch_size=evaluate_batch_size, shuffle=False)

        (avg_e_0, std_e_0), (avg_e_1, std_e_1), (avg_entropy_0, std_entropy_0), (avg_entropy_1, std_entropy_1), (
            avg_ph_dim, std_ph_dim), acc = evalModel(test_loader, model, device)
        avg_e_0_lst.append(avg_e_0)
        avg_e_1_lst.append(avg_e_1)
        avg_entropy_0_lst.append(avg_entropy_0)
        avg_entropy_1_lst.append(avg_entropy_1)
        avg_ph_dim_lst.append(avg_ph_dim)
        acc_list.append(acc)

    # Calculate average and std across classes (by column)
    avg_e_0_output, std_e_0_output = np.average(avg_e_0_lst, axis=0), np.std(avg_e_0_lst, axis=0)
    avg_e_1_output, std_e_1_output = np.average(avg_e_1_lst, axis=0), np.std(avg_e_1_lst, axis=0)
    avg_entropy_0_output, std_entropy_0_output = np.average(avg_entropy_0_lst, axis=0), np.std(avg_entropy_0_lst,
                                                                                               axis=0)
    avg_entropy_1_output, std_entropy_1_output = np.average(avg_entropy_1_lst, axis=0), np.std(avg_entropy_1_lst,
                                                                                               axis=0)
    avg_ph_dim_output, std_ph_dim_output = np.average(avg_ph_dim_lst, axis=0), np.std(avg_ph_dim_lst, axis=0)

    # Save to data
    with open(save_path, 'a') as file:
        for idx in range(len(avg_e_0_output)):
            file.write(
                f"block{idx}, ({avg_e_0_output[idx]}, {std_e_0_output[idx]}), ({avg_e_1_output[idx]}, {std_e_1_output[idx]}),"
                f" ({avg_entropy_0_output[idx]}, {std_entropy_0_output[idx]}), ({avg_entropy_1_output[idx]}, {std_entropy_1_output[idx]}), "
                f" ({avg_ph_dim_output[idx]}, {std_ph_dim_output[idx]})\n")


# Use MPS on Mac if possible
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

path_model = "./results/TrainedModels/AlexNet/AlexNet1.pth"
dataset = "cifar10"
path_save_des = "results/TopologicalDescriptors/AlexNet/CIFAR10_Trained/layers_output.txt"
eval_batch_size = 200
evaluateOutputLayers(path_model, path_save_des, dataset, eval_batch_size, device)

# dataset = "cifar10"
# data_path = "./data/"
# # Get the test dataset
# if dataset in ["cifar10"]:
#     data_class = "CIFAR10"
#     no_class = 10
#     stats = {'mean': [0.491, 0.482, 0.447], 'std' : [0.247, .243, .262]}
#
#     trans = [transforms.ToTensor(),transforms.Normalize(**stats)]
#
#     test_data = getattr(datasets, data_class)(root=data_path,train=False,download=True,transform=transforms.Compose(trans))
#     test_data.data = test_data.data[np.random.choice(test_data.data.shape[0], int(0.2 * test_data.data.shape[0]), replace=False)]
#     dist_matrix = np.zeros((test_data.data.shape[0], test_data.data.shape[0]))
#     for i in range(test_data.data.shape[0]-1):
#         for j in range(test_data.data.shape[0]):
#             dist_matrix[i,j] = np.linalg.norm(test_data.data[i].flatten()/255 - test_data.data[j].flatten()/255)
#     sparse_dist_matrix = triu(dist_matrix)
#     _, _, ph_dim_est, _ = estimatePersistentHomologyDimension(sparse_dist_matrix, 0, 1, 2000, 100, 1000, "precomputed")
#     print(ph_dim_est)


# def evaluate(path, dataset):
#     if dataset in ["cifar10", "cifar10_missing"]:
#         data_class = "CIFAR10"
#         no_class = 10
#         stats = {
#             'mean': [0.491, 0.482, 0.447],
#             'std' : [0.247, .243, .262]
#         }
#     if dataset in ["cifar10"]:
#         trans = [
#             transforms.ToTensor(),
#             transforms.Normalize(**stats)
#         ]
#
#         test_data = getattr(datasets, data_class)(
#             root=path,
#             train=False,
#             download=True,
#             transform=transforms.Compose(trans)
#         )
