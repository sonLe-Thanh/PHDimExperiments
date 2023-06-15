from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PHDimPointCloud import *
from models.AlexNet import AlexNet
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from scipy.spatial.distance import cdist
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance


def evalDataBatch(data_path, save_path, dataset, is_train=False, batch_size=1000, no_neighbors=40, metric="geodesic"):
    """
    Compute the topological descriptors (total life time sum, topological entropy of 0-th, 1-st homology groups, PH_0 dim)
    of a dataset, each values averaging across batches

    :param data_path: path to folder contains dataset, set to "./data" for the current setup
    :param save_path: path to file to saving all these information of the dataset
    :param dataset: name of the dataset, current support: cifar10, mnist
    :param is_train: choose the mode for dataset: True: use train data | False: use test data
    :param batch_size: batch size for each class of the data
    :param no_neighbors: number of neighbors using for the contruction of the k-neighbors graph in geosedic distance
    :param metric: metric use for the computation of distance matrix, current support: geosedic, euclidean
    """
    # Get the test dataset
    if dataset in ["cifar10"]:
        data_class = "CIFAR10"
        no_class = 10
        trans = [transforms.ToTensor()]
        test_data = getattr(datasets, data_class)(root=data_path, train=is_train, download=True,
                                                  transform=transforms.Compose(trans))

    else:
        raise ValueError("Dataset not support atm")
    # # Change stategies
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    e_0_batch_lst = []
    e_1_batch_lst = []
    entropy_0_batch_lst = []
    entropy_1_batch_lst = []
    ph_dim_mean_batch_lst = []
    ph_dim_std_batch_lst = []

    for batch, _ in test_data_loader:
        transformed_data = batch
        batch_size = transformed_data.size(0)

        transformed_data_reshape = transformed_data.view(batch_size, -1).detach().cpu().numpy()
        # Build the distance matrix
        dist_matrix = cdist(transformed_data_reshape, transformed_data_reshape, metric="minkowski")
        if metric == "euclidean":
            e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(dist_matrix, 1, 1.0,
                                                                                     metric="precomputed")
        elif metric == "geodesic":
            # Build the knn graph
            kn_graph_builder = KNeighborsGraph(n_neighbors=no_neighbors, mode='distance', metric='precomputed',
                                               n_jobs=4)
            kn_graph = kn_graph_builder.fit_transform(dist_matrix.reshape(1, *dist_matrix.shape))
            kn_graph = kn_graph[0].todense()
            kn_graph[kn_graph == 0] = np.inf
            for i in range(kn_graph.shape[0]):
                kn_graph[i, i] = 0
            kn_graph = np.squeeze(np.asarray(kn_graph))
            # Get the geodesic distance
            geodesic_dist = \
                GraphGeodesicDistance(directed=False, n_jobs=4).fit_transform(kn_graph.reshape(1, *kn_graph.shape))[0]

            e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(geodesic_dist, 1, 1.0,
                                                                                     metric="precomputed")
        else:
            raise ValueError("Unsupported metric")
        # Add this to the lists
        e_0_batch_lst.append(e_0)
        e_1_batch_lst.append(e_1)
        entropy_0_batch_lst.append(entropy_0)
        entropy_1_batch_lst.append(entropy_1)
        ph_dim_mean_batch_lst.append(ph_dim_info[0])
        ph_dim_std_batch_lst.append(ph_dim_info[1])

    # Average over the classes and save
    e_0_avg, e_0_std = np.average(e_0_batch_lst), np.std(e_0_batch_lst)
    e_1_avg, e_1_std = np.average(e_1_batch_lst), np.std(e_1_batch_lst)
    entropy_0_avg, entropy_0_std = np.average(entropy_0_batch_lst), np.std(entropy_0_batch_lst)
    entropy_1_avg, entropy_1_std = np.average(entropy_1_batch_lst), np.std(entropy_1_batch_lst)
    ph_dim_avg, ph_dim_std = np.average(ph_dim_mean_batch_lst), np.std(ph_dim_std_batch_lst)

    with open(save_path, 'a') as file:
        print("Write file")
        if metric == "geodesic":
            file.write(
                f"{dataset}, {is_train}, {metric}, {no_neighbors}, {batch_size}, ({e_0_avg}, {e_0_std}), ({e_1_avg}, {e_1_std}), ({entropy_0_avg}, {entropy_0_std}), ({entropy_1_avg}, {entropy_0_std}), ({ph_dim_avg}, {ph_dim_std})\n")
        elif metric == "euclidean":
            file.write(
                f"{dataset}, {is_train}, {metric}, {batch_size}, ({e_0_avg}, {e_0_std}), ({e_1_avg}, {e_1_std}), ({entropy_0_avg}, {entropy_0_std}), ({entropy_1_avg}, {entropy_0_std}), ({ph_dim_avg}, {ph_dim_std})\n")



def evalData(data_path, save_path, dataset, is_train=False, batch_size=1000, no_neighbors=40, metric="geodesic"):
    """
    Compute the topological descriptors (total life time sum, topological entropy of 0-th, 1-st homology groups, PH_0 dim)
    of a dataset, each values averaging across batches and then averaging across classes of dataset

    :param data_path: path to folder contains dataset, set to "./data" for the current setup
    :param save_path: path to file to saving all these information of the dataset
    :param dataset: name of the dataset, current support: cifar10, mnist
    :param is_train: choose the mode for dataset: True: use train data | False: use test data
    :param batch_size: batch size for each class of the data
    :param no_neighbors: number of neighbors using for the contruction of the k-neighbors graph in geosedic distance
    :param metric: metric use for the computation of distance matrix, current support: geosedic, euclidean
    """

    # Get the test dataset
    if dataset in ["cifar10"]:
        data_class = "CIFAR10"
        no_class = 10
        trans = [transforms.ToTensor()]
        test_data = getattr(datasets, data_class)(root=data_path, train=is_train, download=True,
                                                  transform=transforms.Compose(trans))

    else:
        raise ValueError("Dataset not support atm")
    e_0_lst = []
    e_1_lst = []
    entropy_0_lst = []
    entropy_1_lst = []
    ph_dim_mean_lst = []
    ph_dim_std_lst = []

    for idx in range(no_class):
        class_idx_list = np.where((np.array(test_data.targets) == idx))[0]
        test_set_1_class = Subset(test_data, class_idx_list)
        test_data_loader = DataLoader(test_set_1_class, batch_size=batch_size, shuffle=False)

        # Reset for each batches
        e_0_batch_lst = []
        e_1_batch_lst = []
        entropy_0_batch_lst = []
        entropy_1_batch_lst = []
        ph_dim_mean_batch_lst = []
        ph_dim_std_batch_lst = []
        for batch, _ in test_data_loader:
            transformed_data = batch
            batch_size = transformed_data.size(0)

            transformed_data_reshape = transformed_data.view(batch_size, -1).detach().cpu().numpy()
            # Build the distance matrix
            dist_matrix = cdist(transformed_data_reshape, transformed_data_reshape, metric="minkowski")
            if metric == "euclidean":
                e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(dist_matrix, 1, 1.0,
                                                                                         metric="precomputed")
            elif metric == "geodesic":
                # Build the knn graph
                kn_graph_builder = KNeighborsGraph(n_neighbors=no_neighbors, mode='distance', metric='precomputed',
                                                   n_jobs=4)
                kn_graph = kn_graph_builder.fit_transform(dist_matrix.reshape(1, *dist_matrix.shape))
                kn_graph = kn_graph[0].todense()
                kn_graph[kn_graph == 0] = np.inf
                for i in range(kn_graph.shape[0]):
                    kn_graph[i, i] = 0
                kn_graph = np.squeeze(np.asarray(kn_graph))
                # Get the geodesic distance
                geodesic_dist = \
                GraphGeodesicDistance(directed=False, n_jobs=4).fit_transform(kn_graph.reshape(1, *kn_graph.shape))[0]

                e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(geodesic_dist, 1, 1.0,
                                                                                         metric="precomputed")
            else:
                raise ValueError("Unsupported metric")
            # Add this to the lists
            e_0_batch_lst.append(e_0)
            e_1_batch_lst.append(e_1)
            entropy_0_batch_lst.append(entropy_0)
            entropy_1_batch_lst.append(entropy_1)
            ph_dim_mean_batch_lst.append(ph_dim_info[0])
            ph_dim_std_batch_lst.append(ph_dim_info[1])

        # Average over the batches and append
        e_0_lst.append(np.average(entropy_0_batch_lst))
        e_1_lst.append(np.average(entropy_1_batch_lst))
        entropy_0_lst.append(np.average(entropy_0_batch_lst))
        entropy_1_lst.append(np.average(entropy_1_batch_lst))
        ph_dim_mean_lst.append(np.average(ph_dim_mean_batch_lst))
        ph_dim_std_lst.append(np.average(ph_dim_std_batch_lst))

    # Average over the classes and save
    e_0_avg, e_0_std = np.average(e_0_lst), np.std(e_0_lst)
    e_1_avg, e_1_std = np.average(e_1_lst), np.std(e_1_lst)
    entropy_0_avg, entropy_0_std = np.average(entropy_0_lst), np.std(entropy_0_lst)
    entropy_1_avg, entropy_1_std = np.average(entropy_1_lst), np.std(entropy_1_lst)
    ph_dim_avg, ph_dim_std = np.average(ph_dim_mean_lst), np.std(ph_dim_mean_lst)

    with open(save_path, 'a') as file:
        print("Write file")
        if metric == "geodesic":
            file.write(f"{dataset}, {is_train}, {metric}, {no_neighbors}, {batch_size}, ({e_0_avg}, {e_0_std}), ({e_1_avg}, {e_1_std}), ({entropy_0_avg}, {entropy_0_std}), ({entropy_1_avg}, {entropy_0_std}), ({ph_dim_avg}, {ph_dim_std})\n")
        elif metric == "euclidean":
            file.write(
                f"{dataset}, {is_train}, {metric}, {batch_size}, ({e_0_avg}, {e_0_std}), ({e_1_avg}, {e_1_std}), ({entropy_0_avg}, {entropy_0_std}), ({entropy_1_avg}, {entropy_0_std}), ({ph_dim_avg}, {ph_dim_std})\n")


def evalModelWeights(weight_path, save_path):
    """
    Compute the topological descriptors (total life time sum, topological entropy of 0-th, 1-st homology groups, PH_0 dim)
    of a flattened model weights

    :param weight_path: full path to weights information, currently only support .npy file
    :param save_path: path to file to saving all these information of the model weights
    """
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
    """
    Helper function for calculating topological descriptors and the accuracy of the model

    :param test_loader: Pytorch test loader for the dataset
    :param model: loader Pytorch model with compatible setup
    :param device: Pytorch device for calculating, default: cpu, possible options: cuda, mds
    :return: total life time sum, topological entropy of 0-th, 1-st homology groups, PH_0 dim, acc;
             All values except for acc whose calculations based on sampling will be return by tuple (mean, std)
    """
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
    else:
        raise ValueError("Model not supported for this function")

    e_0_lst = [[] * 1] * len(return_layers.items())
    e_1_lst = [[] * 1] * len(return_layers.items())
    entropy_0_lst = [[] * 1] * len(return_layers.items())
    entropy_1_lst = [[] * 1] * len(return_layers.items())
    ph_avg_lst = [[] * 1] * len(return_layers.items())
    ph_std_lst = [[] * 1] * len(return_layers.items())
    mid_getter = MidGetter(model, return_layers, keep_output=True)
    print(e_0_lst[0])
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

            for idx, assigned_name in enumerate(mid_outputs):
                output_layer = mid_outputs[assigned_name].to(device).detach().numpy()
                # If tensor has order > 2, reshape
                # if len(output_layer.shape) > 2:
                #     batch_size = output_layer.shape[0]
                #     new_shape = reduce(lambda x, y: x * y, output_layer.shape[1:], 1)
                #     output_layer = output_layer.reshape(batch_size, new_shape)
                print(output_layer.shape)
                # continue
                e_0, e_1, entropy_0, entropy_1, ph_dim_info = computeTopologyDescriptors(
                    output_layer, 1, alpha)
                e_0_lst[idx].append(e_0)
                e_1_lst[idx].append(e_1)
                entropy_1_lst[idx].append(entropy_1)
                entropy_0_lst[idx].append(entropy_0)
                ph_avg_lst[idx].append(ph_dim_info[0])
                ph_std_lst[idx].append(ph_std_lst[0])
    # return
    # Calculate the average and std in each row
    avg_e_0, std_e_0 = np.average(e_0_lst, axis=-1), np.std(e_0_lst, axis=-1)
    avg_e_1, std_e_1 = np.average(e_1_lst, axis=-1), np.std(e_1_lst, axis=-1)
    avg_entropy_0, std_entropy_0 = np.average(entropy_0_lst, axis=-1), np.std(entropy_0_lst, axis=-1)
    avg_entropy_1, std_entropy_1 = np.average(entropy_1_lst, axis=-1), np.std(entropy_1_lst, axis=-1)
    avg_ph_dim, std_ph_dim = np.average(ph_avg_lst, axis=-1), np.std(ph_avg_lst, axis=-1)

    acc = correct / len(test_loader.dataset)

    return (avg_e_0, std_e_0), (avg_e_1, std_e_1), (avg_entropy_0, std_entropy_0), (avg_entropy_1, std_entropy_1), (
        avg_ph_dim, std_ph_dim), acc


def evalOutputLayers(model_path, save_path, data_path, dataset, evaluate_batch_size=1000, device="cpu"):
    """
    Compute the topological descriptors (total life time sum, topological entropy of 0-th, 1-st homology groups, PH_0 dim)
    of the outputs of the intermediate layers of a model, only support AlexNet atm

    :param model_path: full path to saved model state dict trained using Pytorch, currently only support .pth file
    :param save_path: path to file to saving all these information of intermediate layers
    :param data_path: path to folder contains dataset, set to "./data" for the current setup
    :param dataset: name of the dataset, current support: cifar10, mnist
    :param evaluate_batch_size: batch size to feed into each layers
    :param device: Pytorch device for calculating, default: cpu, possible options: cuda, mds
    """
    # Model path
    if "AlexNet" in model_path:
        model = AlexNet(input_height=32, input_width=32, input_channels=3, ch=64, no_class=10)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        raise ValueError("Model not support for evaluating atm")

    # Load test set
    if dataset in ["cifar10"]:
        data_class = "CIFAR10"
        no_class = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, .243, .262]
        }
    else:
        raise ValueError("Dataset not support for evaluating atm")
    trans = [transforms.ToTensor(), transforms.Normalize(**stats)]
    test_data_full = getattr(datasets, data_class)(
        root=data_path,
        download=True,
        train=False,
        transform=transforms.Compose(trans)
    )

    # batch_size = 200
    avg_e_0_lst = []
    avg_e_1_lst = []
    avg_entropy_0_lst = []
    avg_entropy_1_lst = []
    avg_ph_dim_lst = []
    acc_list = []

    # Uncomment for full evaluation
    # for idx in range(no_class):
    for idx in range(0, 1):
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
    avg_acc = np.average(acc_list)

    # Save to data
    with open(save_path, 'a') as file:
        for idx in range(len(avg_e_0_output)):
            file.write(
                f"block{idx + 1}, ({avg_e_0_output[idx]}, {std_e_0_output[idx]}), ({avg_e_1_output[idx]}, {std_e_1_output[idx]}),"
                f" ({avg_entropy_0_output[idx]}, {std_entropy_0_output[idx]}), ({avg_entropy_1_output[idx]}, {std_entropy_1_output[idx]}), "
                f" ({avg_ph_dim_output[idx]}, {std_ph_dim_output[idx]})\n{avg_acc}\n")

if __name__ == "__main__":
    path_data = "./data"
    path_save = "results/TopologicalDescriptors/Datasets/CIFAR10/dataset_batch.txt"
    dataset_name = "cifar10"
    evalDataBatch(path_data, path_save, dataset_name, is_train=False, batch_size=500, no_neighbors=100, metric="geodesic")

    # Note for evaluation on data
    # Datatested: CIFAR10 testset: 10k samples for 10 classes.

    # Use MPS on Mac if possible
    # device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # path_model = "./results/TrainedModels/AlexNet/AlexNet1.pth"
    # dataset = "cifar10"
    # path_save_des = "results/TopologicalDescriptors/AlexNet/CIFAR10_Trained/layers_output.txt"
    # eval_batch_size = 200
    # evaluateOutputLayers(path_model, path_save_des, dataset, eval_batch_size, device)