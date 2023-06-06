import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import math


def typeToText(type_noise:str):
    if type_noise == "FlipPoints":
        return "flipping some weights"
    elif type_noise == "InversePoint":
        return "inverting some weights"
    elif type_noise == "Perturbed":
        return "moving some weights by a random value"
    elif type_noise == "Reduced0":
        return "reducing some weights to 0"
    else:
        return "perturbing"



def drawPHDimTrain(file_name: str):
    full_files = glob.glob(file_name)
    data = genfromtxt(full_files[0], delimiter=',', dtype=str)


    len_file = data.shape[0]
    has_avg = True if "avg" in file_name else False
    model = "AlexNet" if "AlexNet" in file_name else "FC" if "FC" in file_name else ""
    train_acc = []
    test_acc = []
    ph_dim_avg = []
    ph_dim_var = []
    iteration = []
    if model == "AlexNet":
        lr = float(data[0, 0])
        dataset = data[0, 1]
        batch_size = data[0, 2]
        optimizer = data[0, 3]
        for i in range(len_file):
            iteration.append(int(data[i, 4]))
            train_acc.append(3 * float(data[i, 5]))
            test_acc.append(3 * float(data[i, 6]))
            ph_dim_avg.append(float(data[i, 9]))
            if has_avg:
                ph_dim_var.append(float(data[i, 10]))
    elif model == "FC":
        lr = float(data[0, 2])
        dataset = data[0, 3]
        batch_size = data[0, 4]
        optimizer = data[0, 5]
        for i in range(len_file):
            iteration.append(int(data[i, 6]))
            train_acc.append(float(data[i, 7]))
            test_acc.append(float(data[i, 8]))
            ph_dim_avg.append(float(data[i, 11]))
            if has_avg:
                ph_dim_var.append(float(data[i, 12]))
    else:
        return


    gen_error = abs(np.array(train_acc) - np.array(test_acc))
    concate = np.vstack([gen_error, ph_dim_avg, ph_dim_var]).T
    concate = concate[concate[:, 0].argsort()]

    Z = 1.960
    n = 10
    plt.errorbar(concate[:,0], concate[:,1], yerr=concate[:, 2] /2, fmt='go-', ecolor='darkseagreen', elinewidth=0.8, capsize=3)
    plt.scatter(concate[:,0], concate[:,1], c="r", marker="o")

    plt.xticks()
    plt.yticks()
    plt.xlabel("Train acc - test acc (Generalization error)")
    plt.ylabel("dim$_{PH}$")
    plt.title(
        f"PH dimension of AlexNet trained on{dataset} using{optimizer}\nWith batch size{batch_size} and learning rate {lr} Using 2000 previous weights")
    plt.show()
    # plt.savefig("./results/Plots/Models/AlexNet/Avg10Times/GenErrorPHDim_2.png")
    plt.close()

    max_iter = 30000
    eval_every = 1000

    plt.plot(iteration, train_acc, 'b')
    plt.plot(iteration, test_acc, 'r')

    if has_avg:
        ph_dim_var = np.array(ph_dim_var) /2
        plt.errorbar(iteration, ph_dim_avg, yerr=ph_dim_var, fmt='go-', ecolor='darkseagreen', elinewidth=0.8, capsize=3)
    else:
        plt.plot(iteration, ph_dim_avg, 'g')
    plt.xticks()
    plt.yticks()

    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.legend(["Training accuracy", "Testing accuracy", "dim$_{PH}^0$"])
    plt.title(f"PH dimension of AlexNet trained on{dataset} using{optimizer}\nTo 95% accuracy on training set\nWith batch size{batch_size} and learning rate {lr} Using 2000 previous weights")
    plt.show()
    plt.savefig("./results/Plots/Models/AlexNet/TrainTestPHDim_6.png")
    plt.close()

# drawPHDimTrain("results/PHDimReport/PHDimModel/AlexNet_1_avg_5times.txt")
# drawPHDimTrain("results/PHDimReport/PHDimModel/AlexNet_2_avg_10times.txt")
drawPHDimTrain("results/PHDimReport/PHDimModel/AlexNet_4_avg_10times_std.txt")