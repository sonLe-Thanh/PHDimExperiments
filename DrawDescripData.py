import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import math


def drawTopologicalDescriptorData(data_path, save_path):
    """
    Draw from Topological descriptors

    :param data_path: full path to data
    :param save_path: relative path to folder to save the plots
    """

    full_files = glob.glob(data_path)
    data = genfromtxt(full_files[0], delimiter=', ', dtype=str)

    no_records = data.shape[0]

    # Get information on dataset
    dataset_name = data[0,0]
    type_dataset = "Train set" if data[0,1] == True else "Test set"

    # Get information on the calculation


    no_neighbor_lst = []
    # idx 0: geodesic batchsize 1000| 1: geodesic batchsize 500| 2: euclidean
    e_0_mean_lst = [[], [], []]
    e_1_mean_lst = [[], [], []]
    entropy_0_mean_lst = [[], [], []]
    entropy_1_mean_lst = [[], [], []]
    ph_dim_mean_lst = [[], [], []]

    e_0_std_lst = [[], [], []]
    e_1_std_lst = [[], [], []]
    entropy_0_std_lst = [[], [], []]
    entropy_1_std_lst = [[], [], []]
    ph_dim_std_lst = [[], [], []]

    for idx in range(no_records):
        # Read line by line
        metric = data[idx, 2]
        no_neigbor = int(data[idx, 3])
        batch_size = int(data[idx, 4])
        # Only append batchsize when using geodesic and once only
        if metric == "geodesic" and batch_size == 1000:
            no_neighbor_lst.append(no_neigbor)

        # info = (mean, std)
        e_0_info = data[idx, 5][1:-1].split(";")
        e_1_info = data[idx, 6][1:-1].split(";")
        entropy_0_info = data[idx, 7][1:-1].split(";")
        entropy_1_info = data[idx, 8][1:-1].split(";")
        ph_dim_info = data[idx, 9][1:-1].split(";")

        idx_append = 0 if (metric == "geodesic" and batch_size == 1000) else 1 if (metric == "geodesic" and batch_size == 500) else 2

        e_0_mean_lst[idx_append].append(float(e_0_info[0]))
        e_0_std_lst[idx_append].append(float(e_0_info[1]))

        e_1_mean_lst[idx_append].append(float(e_1_info[0]))
        e_1_std_lst[idx_append].append(float(e_1_info[1]))

        entropy_0_mean_lst[idx_append].append(float(entropy_0_info[0]))
        entropy_0_std_lst[idx_append].append(float(entropy_0_info[1]))

        entropy_1_mean_lst[idx_append].append(float(entropy_1_info[0]))
        entropy_1_std_lst[idx_append].append(float(entropy_1_info[1]))

        ph_dim_mean_lst[idx_append].append(float(ph_dim_info[0]))
        ph_dim_std_lst[idx_append].append(float(ph_dim_info[1]))

    # Draw
    # E_0^1
    plt.plot(no_neighbor_lst, e_0_mean_lst[0], 'ro-')
    plt.fill_between(no_neighbor_lst, np.array(e_0_mean_lst[0]) - np.array(e_0_std_lst[0]),
                     np.array(e_0_mean_lst[0]) + np.array(e_0_std_lst[0]), alpha=0.4, facecolor="red")

    plt.plot(no_neighbor_lst, e_0_mean_lst[1], 'bo--')
    plt.fill_between(no_neighbor_lst, np.array(e_0_mean_lst[1]) - np.array(e_0_std_lst[1]),
                          np.array(e_0_mean_lst[1]) + np.array(e_0_std_lst[1]), alpha=0.4, facecolor="blue")

    plt.plot(no_neighbor_lst, [e_0_mean_lst[2][1]] * len(no_neighbor_lst), 'gx-')
    plt.fill_between(no_neighbor_lst, [e_0_mean_lst[2][1] - e_0_std_lst[2][1]] * len(no_neighbor_lst),
                     [e_0_mean_lst[2][1] + e_0_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="green")

    plt.plot(no_neighbor_lst, [e_0_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [e_0_mean_lst[2][0] - e_0_std_lst[2][0]] * len(no_neighbor_lst),
                     [e_0_mean_lst[2][0] + e_0_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("Number of neighbors used")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"Total persistent of 0-th homology group of {dataset_name} {type_dataset}\nUsing Geodesic and Euclidean distance")
    plt.savefig(save_path + "E_0.png")
    plt.show()
    plt.close()


    # E_1^1
    plt.plot(no_neighbor_lst, e_1_mean_lst[0], 'ro-')
    plt.fill_between(no_neighbor_lst, np.array(e_1_mean_lst[0]) - np.array(e_1_std_lst[0]),
                     np.array(e_1_mean_lst[0]) + np.array(e_1_std_lst[0]), alpha=0.4, facecolor="red")

    plt.plot(no_neighbor_lst, e_1_mean_lst[1], 'bo--')
    plt.fill_between(no_neighbor_lst, np.array(e_1_mean_lst[1]) - np.array(e_1_std_lst[1]),
                     np.array(e_1_mean_lst[1]) + np.array(e_1_std_lst[1]), alpha=0.4, facecolor="blue")

    plt.plot(no_neighbor_lst, [e_1_mean_lst[2][1]] * len(no_neighbor_lst), 'gx-')
    plt.fill_between(no_neighbor_lst, [e_1_mean_lst[2][1] - e_1_std_lst[2][1]] * len(no_neighbor_lst),
                     [e_1_mean_lst[2][1] + e_1_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="green")

    plt.plot(no_neighbor_lst, [e_1_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [e_1_mean_lst[2][0] - e_1_std_lst[2][0]] * len(no_neighbor_lst),
                     [e_1_mean_lst[2][0] + e_1_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("Number of neighbors used")
    plt.ylabel("$E_1^1$")
    plt.title(
        f"Total persistent of 1-st homology group of {dataset_name} {type_dataset}\nUsing Geodesic and Euclidean distance")
    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
    plt.savefig(save_path + "E_1.png")
    plt.show()
    plt.close()

    # entropy_0
    plt.plot(no_neighbor_lst, entropy_0_mean_lst[0], 'ro-')
    plt.fill_between(no_neighbor_lst, np.array(entropy_0_mean_lst[0]) - np.array(entropy_0_std_lst[0]),
                     np.array(entropy_0_mean_lst[0]) + np.array(entropy_0_std_lst[0]), alpha=0.4, facecolor="red")

    plt.plot(no_neighbor_lst, entropy_0_mean_lst[1], 'bo--')
    plt.fill_between(no_neighbor_lst, np.array(entropy_0_mean_lst[1]) - np.array(entropy_0_std_lst[1]),
                     np.array(entropy_0_mean_lst[1]) + np.array(entropy_0_std_lst[1]), alpha=0.4, facecolor="blue")

    plt.plot(no_neighbor_lst, [entropy_0_mean_lst[2][1]] * len(no_neighbor_lst), 'gx-')
    plt.fill_between(no_neighbor_lst, [entropy_0_mean_lst[2][1] - entropy_0_std_lst[2][1]] * len(no_neighbor_lst),
                     [entropy_0_mean_lst[2][1] + entropy_0_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="green")

    plt.plot(no_neighbor_lst, [entropy_0_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [entropy_0_mean_lst[2][0] - entropy_0_std_lst[2][0]] * len(no_neighbor_lst),
                     [entropy_0_mean_lst[2][0] + entropy_0_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("Number of neighbors used")
    plt.ylabel("$entropy_0$")
    plt.title(
        f"Topological entropy of 0-th homology group of {dataset_name} {type_dataset}\nUsing Geodesic and Euclidean distance")
    plt.savefig(save_path + "entropy_0.png")
    plt.show()
    plt.close()

    # entropy_1
    plt.plot(no_neighbor_lst, entropy_1_mean_lst[0], 'ro-')
    plt.fill_between(no_neighbor_lst, np.array(entropy_1_mean_lst[0]) - np.array(entropy_1_std_lst[0]),
                     np.array(entropy_1_mean_lst[0]) + np.array(entropy_1_std_lst[0]), alpha=0.4, facecolor="red")

    plt.plot(no_neighbor_lst, entropy_1_mean_lst[1], 'bo--')
    plt.fill_between(no_neighbor_lst, np.array(entropy_1_mean_lst[1]) - np.array(entropy_1_std_lst[1]),
                     np.array(entropy_1_mean_lst[1]) + np.array(entropy_1_std_lst[1]), alpha=0.4, facecolor="blue")

    plt.plot(no_neighbor_lst, [entropy_1_mean_lst[2][1]] * len(no_neighbor_lst), 'gx-')
    plt.fill_between(no_neighbor_lst, [entropy_1_mean_lst[2][1] - entropy_1_std_lst[2][1]] * len(no_neighbor_lst),
                     [entropy_1_mean_lst[2][1] + entropy_1_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="green")

    plt.plot(no_neighbor_lst, [entropy_1_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [entropy_1_mean_lst[2][0] - entropy_1_std_lst[2][0]] * len(no_neighbor_lst),
                     [entropy_1_mean_lst[2][0] + entropy_1_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("Number of neighbors used")
    plt.ylabel("$entropy_1$")
    plt.title(
        f"Topological entropy of 1-st homology group of {dataset_name} {type_dataset}\nUsing Geodesic and Euclidean distance")
    plt.savefig(save_path + "entropy_1.png")
    plt.show()
    plt.close()

    # PH dim
    plt.plot(no_neighbor_lst, ph_dim_mean_lst[0], 'ro-')
    plt.fill_between(no_neighbor_lst, np.array(ph_dim_mean_lst[0]) - np.array(ph_dim_std_lst[0]),
                     np.array(ph_dim_mean_lst[0]) + np.array(ph_dim_std_lst[0]), alpha=0.4, facecolor="red")

    plt.plot(no_neighbor_lst, ph_dim_mean_lst[1], 'bo--')
    plt.fill_between(no_neighbor_lst, np.array(ph_dim_mean_lst[1]) - np.array(ph_dim_std_lst[1]),
                     np.array(ph_dim_mean_lst[1]) + np.array(ph_dim_std_lst[1]), alpha=0.4, facecolor="blue")

    plt.plot(no_neighbor_lst, [ph_dim_mean_lst[2][1]] * len(no_neighbor_lst), 'gx-')
    plt.fill_between(no_neighbor_lst, [ph_dim_mean_lst[2][1] - ph_dim_std_lst[2][1]] * len(no_neighbor_lst),
                     [ph_dim_mean_lst[2][1] + ph_dim_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4, facecolor="green")

    plt.plot(no_neighbor_lst, [ph_dim_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [ph_dim_mean_lst[2][0] - ph_dim_std_lst[2][0]] * len(no_neighbor_lst),
                     [ph_dim_mean_lst[2][0] + ph_dim_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500" , "Euclidean batch size 1000", "Euclidean batch size 500"])
    plt.ylim(0, 30)
    plt.xticks()
    plt.yticks()
    plt.xlabel("Number of neighbors used")
    plt.ylabel("dim$_{PH}$")
    plt.title(
        f"PH dimension of {dataset_name} {type_dataset} using Geodesic and Euclidean distance")
    plt.savefig(save_path + "PH_dim.png")
    plt.show()
    plt.close()



path_res = "./results/TopologicalDescriptors/Datasets/CIFAR10/dataset_batch.txt"
path_save = "./results/Plots/TopologicalDescriptors/Dataset/CIFAR10/Batch/"
drawTopologicalDescriptorData(path_res, path_save)