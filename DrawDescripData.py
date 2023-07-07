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
    dataset_name = data[0, 0]
    type_dataset = "Train set" if data[0, 1] == True else "Test set"

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

        idx_append = 0 if (metric == "geodesic" and batch_size == 1000) else 1 if (
                    metric == "geodesic" and batch_size == 500) else 2

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
                     [ph_dim_mean_lst[2][1] + ph_dim_std_lst[2][1]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="green")

    plt.plot(no_neighbor_lst, [ph_dim_mean_lst[2][0]] * len(no_neighbor_lst), 'yx--')
    plt.fill_between(no_neighbor_lst, [ph_dim_mean_lst[2][0] - ph_dim_std_lst[2][0]] * len(no_neighbor_lst),
                     [ph_dim_mean_lst[2][0] + ph_dim_std_lst[2][0]] * len(no_neighbor_lst), alpha=0.4,
                     facecolor="yellow")

    plt.legend(["Geodesic batch size 1000", "Geodesic batch size 500", "Euclidean batch size 1000",
                "Euclidean batch size 500"])
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

def drawTopologicalDescriptorAdversarialData2Types(data_path, save_path):
    """
    Draw from Topological descriptors from 2 types of noise

    :param data_path: full path to adversarial data information
    :param save_path: relative path to folder to save the plots
    """

    full_files = glob.glob(data_path)
    data = genfromtxt(full_files[0], delimiter=', ', dtype=str)

    no_records = data.shape[0]
    dataset_name = data[0, 0].split("_")[0]
    model_name = data[0, 0].split("_")[3]
    type_dataset = "Train set" if data[0, 1] is True else "Test set"

    # Get information on the calculation
    # 0: fgsm attack, 1: pgd attack
    eps_lst = []
    acc_lst = [[], []]
    e_0_mean_lst = [[], []]
    e_1_mean_lst = [[], []]
    ph_dim_mean_lst = [[], []]

    e_0_std_lst = [[], []]
    e_1_std_lst = [[], []]
    ph_dim_std_lst = [[], []]

    for idx in range(no_records):
        # Read line by line
        # Get information on dataset
        dataset_info = data[idx, 0].split("_")
        type_attack = dataset_info[1]
        attack_param = float(dataset_info[2])
        accuracy = float(dataset_info[4].split(":")[1])

        idx_append = 0 if type_attack == "fgsm" else 1
        if idx_append == 0:
            eps_lst.append(attack_param)

        acc_lst[idx_append].append(accuracy)

        e_0_info = data[idx, 5][1:-1].split(";")
        e_1_info = data[idx, 6][1:-1].split(";")

        ph_dim_info = data[idx, 9][1:-1].split(";")

        e_0_mean_lst[idx_append].append(float(e_0_info[0]))
        e_0_std_lst[idx_append].append(float(e_0_info[1]))

        e_1_mean_lst[idx_append].append(float(e_1_info[0]))
        e_1_std_lst[idx_append].append(float(e_1_info[1]))

        ph_dim_mean_lst[idx_append].append(float(ph_dim_info[0]))
        ph_dim_std_lst[idx_append].append(float(ph_dim_info[1]))

    # Draw
    # acc
    plt.plot(eps_lst, acc_lst[0], 'ro-')
    plt.plot(eps_lst, acc_lst[1], 'bo-')

    plt.legend(["Fast gradient sign", "Projected gradient descent"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy of {model_name} tested by {dataset_name} under adversarial attack")
    plt.savefig(save_path + "acc.png")
    plt.show()
    plt.close()

    # E_0^1
    plt.plot(eps_lst, e_0_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_0_mean_lst[1], 'bo-')


    plt.fill_between(eps_lst, np.array(e_0_mean_lst[0]) - np.array(e_0_std_lst[0]),
                     np.array(e_0_mean_lst[0]) + np.array(e_0_std_lst[0]), alpha=0.3, facecolor="red")
    plt.fill_between(eps_lst, np.array(e_0_mean_lst[1]) - np.array(e_0_std_lst[1]),
                     np.array(e_0_mean_lst[1]) + np.array(e_0_std_lst[1]), alpha=0.3, facecolor="blue")


    plt.legend(["Fast gradient sign", "Projected gradient descent"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"Total lifetime sum of 0-th homology group of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "E_0.png")
    plt.show()
    plt.close()

    # E_1^1
    plt.plot(eps_lst, e_1_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_1_mean_lst[1], 'bo-')

    plt.fill_between(eps_lst, np.array(e_1_mean_lst[0]) - np.array(e_1_std_lst[0]),
                     np.array(e_1_mean_lst[0]) + np.array(e_1_std_lst[0]), alpha=0.3, facecolor="red")

    plt.fill_between(eps_lst, np.array(e_1_mean_lst[1]) - np.array(e_1_std_lst[1]),
                     np.array(e_1_mean_lst[1]) + np.array(e_1_std_lst[1]), alpha=0.3, facecolor="blue")

    plt.legend(["Fast gradient sign", "Projected gradient descent"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_1^1$")
    plt.title(
         f"Total lifetime sum of 1-st homology group of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "E_1.png")
    plt.show()
    plt.close()


    # # PH dim
    plt.plot(eps_lst, ph_dim_mean_lst[0], 'ro-')
    plt.plot(eps_lst, ph_dim_mean_lst[1], 'bo-')

    plt.fill_between(eps_lst, np.array(ph_dim_mean_lst[0]) - np.array(ph_dim_std_lst[0]),
                     np.array(ph_dim_mean_lst[0]) + np.array(ph_dim_std_lst[0]), alpha=0.3, facecolor="red")
    plt.fill_between(eps_lst, np.array(ph_dim_mean_lst[1]) - np.array(ph_dim_std_lst[1]),
                     np.array(ph_dim_mean_lst[1]) + np.array(ph_dim_std_lst[1]), alpha=0.3, facecolor="blue")


    plt.legend(["Fast gradient sign", "Projected gradient descent"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("PH$_dim$")
    plt.title(
         f"PH dimension of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "PH_dim.png")
    plt.show()
    plt.close()


def drawTopologicalDescriptorAdversarialData3Types(data_path, save_path):
    """
    Draw from Topological descriptors from 3 types of noise

    :param data_path: full path to adversarial data information
    :param save_path: relative path to folder to save the plots
    """

    full_files = glob.glob(data_path)
    data = genfromtxt(full_files[0], delimiter=', ', dtype=str)

    no_records = data.shape[0]
    dataset_name = data[0, 0].split("_")[0]
    model_name = data[0, 0].split("_")[3]
    type_dataset = "Train set" if data[0, 1] is True else "Test set"

    # Get information on the calculation
    # 0: fgsm attack, 1: pgd attack, 2: random attack
    eps_lst = []
    acc_lst = [[], [], []]
    e_0_mean_lst = [[], [], []]
    e_1_mean_lst = [[], [], []]
    ph_dim_mean_lst = [[], [], []]

    e_0_std_lst = [[], [], []]
    e_1_std_lst = [[], [], []]
    ph_dim_std_lst = [[], [], []]

    for idx in range(no_records):
        # Read line by line
        # Get information on dataset
        dataset_info = data[idx, 0].split("_")
        type_attack = dataset_info[1]
        attack_param = float(dataset_info[2])
        accuracy = float(dataset_info[4].split(":")[1])

        idx_append = 0 if type_attack == "fgsm" else 1 if type_attack == "pgd" else 2
        if idx_append == 0:
            eps_lst.append(attack_param)

        acc_lst[idx_append].append(accuracy)

        e_0_info = data[idx, 5][1:-1].split(";")
        e_1_info = data[idx, 6][1:-1].split(";")

        ph_dim_info = data[idx, 9][1:-1].split(";")

        e_0_mean_lst[idx_append].append(float(e_0_info[0]))
        e_0_std_lst[idx_append].append(float(e_0_info[1]))

        e_1_mean_lst[idx_append].append(float(e_1_info[0]))
        e_1_std_lst[idx_append].append(float(e_1_info[1]))

        ph_dim_mean_lst[idx_append].append(float(ph_dim_info[0]))
        ph_dim_std_lst[idx_append].append(float(ph_dim_info[1]))

    # Draw
    # acc
    plt.plot(eps_lst, acc_lst[0], 'ro-')
    plt.plot(eps_lst, acc_lst[1], 'bo-')
    plt.plot(eps_lst, acc_lst[2], 'go-')

    plt.legend(["Fast gradient sign", "Projected gradient descent", "Random noise"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy of {model_name} tested by {dataset_name} under adversarial attack")
    plt.savefig(save_path + "acc.png")
    plt.show()
    plt.close()

    # E_0^1
    plt.plot(eps_lst, e_0_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_0_mean_lst[1], 'bo-')
    plt.plot(eps_lst, e_0_mean_lst[2], 'go-')


    plt.fill_between(eps_lst, np.array(e_0_mean_lst[0]) - np.array(e_0_std_lst[0]),
                     np.array(e_0_mean_lst[0]) + np.array(e_0_std_lst[0]), alpha=0.3, facecolor="red")
    plt.fill_between(eps_lst, np.array(e_0_mean_lst[1]) - np.array(e_0_std_lst[1]),
                     np.array(e_0_mean_lst[1]) + np.array(e_0_std_lst[1]), alpha=0.3, facecolor="blue")
    plt.fill_between(eps_lst, np.array(e_0_mean_lst[2]) - np.array(e_0_std_lst[2]),
                     np.array(e_0_mean_lst[2]) + np.array(e_0_std_lst[2]), alpha=0.3, facecolor="green")

    plt.legend(["Fast gradient sign", "Projected gradient descent", "Random noise"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"Total lifetime sum of 0-th homology group of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "E_0.png")
    plt.show()
    plt.close()

    # E_1^1
    plt.plot(eps_lst, e_1_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_1_mean_lst[1], 'bo-')
    plt.plot(eps_lst, e_1_mean_lst[2], 'go-')
    plt.fill_between(eps_lst, np.array(e_1_mean_lst[0]) - np.array(e_1_std_lst[0]),
                     np.array(e_1_mean_lst[0]) + np.array(e_1_std_lst[0]), alpha=0.3, facecolor="red")

    plt.fill_between(eps_lst, np.array(e_1_mean_lst[1]) - np.array(e_1_std_lst[1]),
                     np.array(e_1_mean_lst[1]) + np.array(e_1_std_lst[1]), alpha=0.3, facecolor="blue")
    plt.fill_between(eps_lst, np.array(e_1_mean_lst[2]) - np.array(e_1_std_lst[2]),
                     np.array(e_1_mean_lst[2]) + np.array(e_1_std_lst[2]), alpha=0.3, facecolor="green")
    plt.legend(["Fast gradient sign", "Projected gradient descent", "Random noise"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_1^1$")
    plt.title(
         f"Total lifetime sum of 1-st homology group of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "E_1.png")
    plt.show()
    plt.close()


    # # PH dim
    plt.plot(eps_lst, ph_dim_mean_lst[0], 'ro-')
    plt.plot(eps_lst, ph_dim_mean_lst[1], 'bo-')
    plt.plot(eps_lst, ph_dim_mean_lst[2], 'go-')
    plt.fill_between(eps_lst, np.array(ph_dim_mean_lst[0]) - np.array(ph_dim_std_lst[0]),
                     np.array(ph_dim_mean_lst[0]) + np.array(ph_dim_std_lst[0]), alpha=0.3, facecolor="red")
    plt.fill_between(eps_lst, np.array(ph_dim_mean_lst[1]) - np.array(ph_dim_std_lst[1]),
                     np.array(ph_dim_mean_lst[1]) + np.array(ph_dim_std_lst[1]), alpha=0.3, facecolor="blue")
    plt.fill_between(eps_lst, np.array(ph_dim_mean_lst[2]) - np.array(ph_dim_std_lst[2]),
                     np.array(ph_dim_mean_lst[2]) + np.array(ph_dim_std_lst[2]), alpha=0.3, facecolor="green")

    plt.legend(["Fast gradient sign", "Projected gradient descent", "Random noise"])
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("PH$_dim$")
    plt.title(
         f"PH dimension of adversarial {dataset_name.upper()}\nCreated by {model_name} and Cross Entropy Loss")
    plt.savefig(save_path + "PH_dim.png")
    plt.show()
    plt.close()

def drawTopologicalDescriptorAcrossDataClass(data_path, save_path):
    """
    Draw from Topological descriptors

    :param data_path: full path to across data class information
    :param save_path: relative path to folder to save the plots
    """

    full_files = glob.glob(data_path)
    data = genfromtxt(full_files[0], delimiter=', ', dtype=str)

    no_records = data.shape[0]
    dataset_name = data[0, 0].split("_")[0]
    type_dataset = "Train set" if data[0, 1] is True else "Test set"

    # Get information on the calculation
    class_idx = []
    e_0_mean_lst = []
    e_1_mean_lst = []
    ph_dim_mean_lst = []

    e_0_std_lst = []
    e_1_std_lst = []
    ph_dim_std_lst = []
    e_0_mean = 0
    e_0_std = 0
    e_1_mean = 0
    e_1_std = 0
    ph_dim_mean = 0
    ph_dim_std = 0
    for idx in range(no_records):
        # Read line by line
        e_0_info = data[idx, 5][1:-1].split(";")
        e_1_info = data[idx, 6][1:-1].split(";")

        ph_dim_info = data[idx, 9][1:-1].split(";")
        # Get information on dataset
        dataset_info = data[idx, 0].split("_")
        if len(dataset_info) > 1:
            class_idx.append(int(dataset_info[1][-1]))
            e_0_mean_lst.append(float(e_0_info[0]))
            e_0_std_lst.append(float(e_0_info[1]))

            e_1_mean_lst.append(float(e_1_info[0]))
            e_1_std_lst.append(float(e_1_info[1]))

            ph_dim_mean_lst.append(float(ph_dim_info[0]))
            ph_dim_std_lst.append(float(ph_dim_info[1]))
        else:
            # The average over dataset
            e_0_mean = float(e_0_info[0])
            e_0_std = float(e_0_info[1])
            e_1_mean = float(e_1_info[0])
            e_1_std = float(e_1_info[1])
            ph_dim_mean = float(ph_dim_info[0])
            ph_dim_std = float(ph_dim_info[1])

    # Draw
    # E_0^1
    plt.plot(class_idx, [e_0_mean] * len(class_idx), 'r-')
    plt.scatter(class_idx, e_0_mean_lst, c='b', marker='o')
    plt.fill_between(class_idx, [e_0_mean - e_0_std] * len(class_idx),
                     [e_0_mean + e_0_std] * len(class_idx), alpha=0.3, facecolor="red")

    plt.xticks(np.arange(0, int(class_idx[-1])+1, 1.0))
    plt.yticks()
    plt.xlabel("Class")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"Total lifetime sum of 0-th homology group of classes of {dataset_name.upper()}")
    plt.legend(["Average"])
    plt.savefig(save_path + "E_0.png")
    plt.show()
    plt.close()

    # E_1^1
    plt.plot(class_idx, [e_1_mean] * len(class_idx), 'r-')
    plt.scatter(class_idx, e_1_mean_lst, c='b', marker='o')
    plt.fill_between(class_idx, [e_1_mean - e_1_std] * len(class_idx),
                     [e_1_mean + e_1_std] * len(class_idx), alpha=0.3, facecolor="red")

    plt.xticks(np.arange(0, int(class_idx[-1])+1, 1.0))
    plt.yticks()
    plt.xlabel("Class")
    plt.ylabel("$E_1^1$")
    plt.title(
        f"Total lifetime sum of 1-st homology group of classes of {dataset_name.upper()}")
    plt.legend(["Average"])
    plt.savefig(save_path + "E_1.png")
    plt.show()
    plt.close()

    # PH dim
    plt.plot(class_idx, [ph_dim_mean] * len(class_idx), 'r-')
    plt.errorbar(class_idx, ph_dim_mean_lst, yerr=ph_dim_std_lst, fmt="bo", elinewidth=0.5)
    # plt.scatter(class_idx, ph_dim_mean_lst, c='b', marker='o')
    plt.fill_between(class_idx, [ph_dim_mean - ph_dim_std] * len(class_idx),
                     [ph_dim_mean + ph_dim_std] * len(class_idx), alpha=0.3, facecolor="red")

    plt.xticks(np.arange(0, int(class_idx[-1])+1, 1.0))
    plt.yticks()
    plt.xlabel("Class")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"PH dim of classes of {dataset_name.upper()}")
    plt.legend(["Average"])
    plt.savefig(save_path + "PH_dim.png")
    plt.show()
    plt.close()


def drawTopologicalDescriptorAdversarialDataClass(data_path, save_path):
    """
    Draw from Topological descriptors

    :param data_path: full path to adversarial data information
    :param save_path: relative path to folder to save the plots
    """

    full_files = glob.glob(data_path)
    data = genfromtxt(full_files[0], delimiter=', ', dtype=str)

    no_records = data.shape[0]
    dataset_name = data[0, 0].split("_")[0]
    model_name = data[0, 0].split("_")[3]
    type_dataset = "Train set" if data[0, 1] is True else "Test set"

    # Get information on the calculation
    #
    eps_lst = []
    acc_lst = [[] for _ in range(10)]
    e_0_mean_lst = [[] for _ in range(10)]
    e_1_mean_lst = [[] for _ in range(10)]
    ph_dim_mean_lst = [[] for _ in range(10)]

    e_0_std_lst = [[] for _ in range(10)]
    e_1_std_lst = [[] for _ in range(10)]
    ph_dim_std_lst = [[] for _ in range(10)]

    for idx in range(no_records):
        # Read line by line
        # Get information on dataset
        dataset_info = data[idx, 0].split("_")
        type_attack = dataset_info[1]
        attack_param = float(dataset_info[2])
        class_idx = int(dataset_info[3][-1])
        accuracy = float(dataset_info[4].split(":")[1])

        if class_idx == 0:
            eps_lst.append(attack_param)

        acc_lst[class_idx].append(accuracy)

        e_0_info = data[idx, 5][1:-1].split(";")
        e_1_info = data[idx, 6][1:-1].split(";")

        ph_dim_info = data[idx, 9][1:-1].split(";")

        e_0_mean_lst[class_idx].append(float(e_0_info[0]))
        e_0_std_lst[class_idx].append(float(e_0_info[1]))

        e_1_mean_lst[class_idx].append(float(e_1_info[0]))
        e_1_std_lst[class_idx].append(float(e_1_info[1]))

        ph_dim_mean_lst[class_idx].append(float(ph_dim_info[0]))
        ph_dim_std_lst[class_idx].append(float(ph_dim_info[1]))

    # Draw
    # acc
    plt.plot(eps_lst, acc_lst[0], 'ro-')
    plt.plot(eps_lst, acc_lst[1], 'bo-')
    plt.plot(eps_lst, acc_lst[2], 'go-')
    plt.plot(eps_lst, acc_lst[3], 'co-')
    plt.plot(eps_lst, acc_lst[4], 'mo-')
    plt.plot(eps_lst, acc_lst[5], 'ro--')
    plt.plot(eps_lst, acc_lst[6], 'bo--')
    plt.plot(eps_lst, acc_lst[7], 'go--')
    plt.plot(eps_lst, acc_lst[8], 'co--')
    plt.plot(eps_lst, acc_lst[9], 'mo--')
    label_name = ["Class"+str(i) for i in range(10)]
    plt.legend(label_name)
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy of {model_name[:-6]} tested by {dataset_name} under adversarial attack")
    plt.savefig(save_path + "acc.png")
    plt.show()
    plt.close()

    # E_0^1
    plt.plot(eps_lst, e_0_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_0_mean_lst[1], 'bo-')
    plt.plot(eps_lst, e_0_mean_lst[2], 'go-')
    plt.plot(eps_lst, e_0_mean_lst[3], 'co-')
    plt.plot(eps_lst, e_0_mean_lst[4], 'mo-')
    plt.plot(eps_lst, e_0_mean_lst[5], 'ro--')
    plt.plot(eps_lst, e_0_mean_lst[6], 'bo--')
    plt.plot(eps_lst, e_0_mean_lst[7], 'go--')
    plt.plot(eps_lst, e_0_mean_lst[8], 'co--')
    plt.plot(eps_lst, e_0_mean_lst[9], 'mo--')

    label_name = ["Class" + str(i) for i in range(10)]
    plt.legend(label_name)
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_0^1$")
    plt.title(
        f"Total lifetime sum of 0-th homology group of adversarial {dataset_name.upper()}\nCreated by {model_name[:-6]} and Cross Entropy Loss")
    plt.savefig(save_path + "E_0.png")
    plt.show()
    plt.close()

    # E_1^1
    plt.plot(eps_lst, e_1_mean_lst[0], 'ro-')
    plt.plot(eps_lst, e_1_mean_lst[1], 'bo-')
    plt.plot(eps_lst, e_1_mean_lst[2], 'go-')
    plt.plot(eps_lst, e_1_mean_lst[3], 'co-')
    plt.plot(eps_lst, e_1_mean_lst[4], 'mo-')
    plt.plot(eps_lst, e_1_mean_lst[5], 'ro--')
    plt.plot(eps_lst, e_1_mean_lst[6], 'bo--')
    plt.plot(eps_lst, e_1_mean_lst[7], 'go--')
    plt.plot(eps_lst, e_1_mean_lst[8], 'co--')
    plt.plot(eps_lst, e_1_mean_lst[9], 'mo--')

    label_name = ["Class" + str(i) for i in range(10)]
    plt.legend(label_name)
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$E_1^1$")
    plt.title(
         f"Total lifetime sum of 1-st homology group of adversarial {dataset_name.upper()}\nCreated by {model_name[:-6]} and Cross Entropy Loss")
    plt.savefig(save_path + "E_1.png")
    plt.show()
    plt.close()


    # # PH dim
    plt.plot(eps_lst, ph_dim_mean_lst[0], 'ro-')
    plt.plot(eps_lst, ph_dim_mean_lst[1], 'bo-')
    plt.plot(eps_lst, ph_dim_mean_lst[2], 'go-')
    plt.plot(eps_lst, ph_dim_mean_lst[3], 'co-')
    plt.plot(eps_lst, ph_dim_mean_lst[4], 'mo-')
    plt.plot(eps_lst, ph_dim_mean_lst[5], 'ro--')
    plt.plot(eps_lst, ph_dim_mean_lst[6], 'bo--')
    plt.plot(eps_lst, ph_dim_mean_lst[7], 'go--')
    plt.plot(eps_lst, ph_dim_mean_lst[8], 'co--')
    plt.plot(eps_lst, ph_dim_mean_lst[9], 'mo--')

    label_name = ["Class" + str(i) for i in range(10)]
    plt.legend(label_name)
    plt.xticks()
    plt.yticks()
    plt.xlabel("$\epsilon$")
    plt.ylabel("PH$_dim$")
    plt.title(
         f"PH dimension of adversarial {dataset_name.upper()}\nCreated by {model_name[:-6]} and Cross Entropy Loss")
    plt.savefig(save_path + "PH_dim.png")
    plt.show()
    plt.close()


path_res = "results/TopologicalDescriptors/Datasets/MNIST/TestData/dataset_class_attack_adversarial_training.txt"
path_save = "./results/Plots/TopologicalDescriptors/Dataset/MNIST/AdversarialAttackClass/AdversarialTraining/"
drawTopologicalDescriptorAdversarialDataClass(path_res, path_save)