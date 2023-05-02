import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


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



def drawFromPertubation(type_noise:str, model_name:str):
    full_files = glob.glob('results/PHDimReport/PHDimModelWeights/'+type_noise+'/'+model_name+".txt")
    data = genfromtxt(full_files[0], delimiter=',', dtype=str)
    trained_param = np.array(list(map(lambda x: x.split('|'), data[:,0])))

    trained_param[:,2] = np.array(list(map(lambda x: x[10:], trained_param[:,2])))
    trained_param[:,3] = np.array(list(map(lambda x: x[4:], trained_param[:,3])))
    trained_param[:,4] = np.array(list(map(lambda x: x[3:], trained_param[:,4])))

    len_file = data.shape[0]


    cifar10_100_SGD_01_0 = []
    cifar10_200_SGD_01_0 = []
    mnist_100_SGD_01_0 = []
    mnist_200_Adam_005_0 = []
    mnist_200_SGD_01_0 = []
    mnist_500_Adam_001_0 = []

    cifar10_100_SGD_01_1 = []
    cifar10_200_SGD_01_1 = []
    mnist_100_SGD_01_1 = []
    mnist_200_Adam_005_1 = []
    mnist_200_SGD_01_1 = []
    mnist_500_Adam_001_1 = []

    print(data[:,3])
    # return


    for i in range(len_file):
        if trained_param[i, 1] == "cifar10":
            if trained_param[i, 2] == "100":
                if trained_param[i, 3] == "Adam":
                    if trained_param[i, 4] == "0.1": pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass
                elif trained_param[i, 3] == "SGD":
                    if trained_param[i, 4] == "0.1":
                        if data[i, 3] == "0":
                            cifar10_100_SGD_01_0.append(data[i, 5])
                        else:
                            pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass
            elif trained_param[i, 2] == "200":
                if trained_param[i, 3] == "Adam":
                    if trained_param[i, 4] == "0.1": pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass
                elif trained_param[i, 3] == "SGD":
                    if trained_param[i, 4] == "0.1":
                        if data[i, 3] == "0":
                            cifar10_200_SGD_01_0.append(data[i, 5])
                        else:
                            pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass

        elif trained_param[i, 1] == "mnist":
            if trained_param[i, 2] == "100":
                if trained_param[i, 3] == "Adam":
                    if trained_param[i, 4] == "0.1": pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass
                elif trained_param[i, 3] == "SGD":
                    if trained_param[i, 4] == "0.1":
                        if data[i, 3] == "0":
                            mnist_200_Adam_005_0.append(data[i, 5])
                        else:
                            pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05": pass

            elif trained_param[i, 2] == "200":
                if trained_param[i, 3] == "Adam":
                    if trained_param[i, 4] == "0.1": pass
                    elif trained_param[i,4] == "0.01": pass
                    elif trained_param[i, 4] == "0.05":
                        if trained_param[i, 4] == "0.1":
                            if data[i, 3] == "0":
                                cifar10_200_SGD_01_0.append(data[i, 5])
                            else:
                                pass
            #     elif trained_param[i, 3] == "SGD":
            #         if trained_param[i, 4] == "0.1":
            #         elif trained_param[i,4] == "0.01":
            #         elif trained_param[i, 4] == "0.05":
            #
            # elif trained_param[i, 2] == "500":
            #     if trained_param[i, 3] == "Adam":
            #         if trained_param[i, 4] == "0.1":
            #         elif trained_param[i,4] == "0.01":
            #         elif trained_param[i, 4] == "0.05":
            #     elif trained_param[i, 3] == "SGD":
            #         if trained_param[i, 4] == "0.1":
            #         elif trained_param[i,4] == "0.01":
            #         elif trained_param[i, 4] == "0.05":
        else:
            pass
    # break
        # data_type = full_path.split('/')[4].split('.')[0]
        #
        # data = genfromtxt(full_path, delimiter=':', dtype=str)
        # data = np.concatenate((np.delete(data, 1 , -1), np.array(list(map(lambda x: x.split(','), data[:,1])))), axis=1)
        #
        # no_data_point = int(data[0, 1])
        #
        # percentage_noise_0_dim = []
        # percentage_noise_1_dim = []
        #
        # ph_dim_0_dim_true = []
        # ph_dim_1_dim_true = []
        #
        # ph_dim_0_dim_mutated = []
        # ph_dim_1_dim_mutated = []
        #
        # for i in range(data.shape[0]):
        #     if data[i, 0] == "True":
        #         if int(data[i, 3]) == 0:
        #             ph_dim_0_dim_true.append(float(data[i, 4]))
        #         elif int(data[i, 3]) == 1:
        #             ph_dim_1_dim_true.append(float(data[i, 4]))
        #         else:
        #             raise ValueError("Not recognizing value")
        #     elif data[i, 0] == "Mutated":
        #         if int(data[i, 3]) == 0:
        #             ph_dim_0_dim_mutated.append(float(data[i, 4]))
        #             percentage_noise_0_dim.append(float(data[i, 2]))
        #         elif int(data[i, 3]) == 1:
        #             ph_dim_1_dim_mutated.append(float(data[i, 4]))
        #             percentage_noise_1_dim.append(float(data[i, 2]))
        #         else:
        #             raise ValueError("Not recognizing value")
        #     else:
        #         raise ValueError("Not recognizing value")
        #
        # plt.subplots_adjust(right=0.7)
        # plt.plot(percentage_noise_0_dim, ph_dim_0_dim_true, 'b')
        # plt.plot(percentage_noise_0_dim, ph_dim_0_dim_mutated, '--b')
        # plt.plot(percentage_noise_0_dim, ph_dim_1_dim_true, 'r')
        # plt.plot(percentage_noise_0_dim, ph_dim_1_dim_mutated, '--r')
        #
        # plt.scatter(percentage_noise_0_dim, ph_dim_0_dim_true, c="b", marker="o")
        # plt.scatter(percentage_noise_0_dim, ph_dim_0_dim_mutated, c="b", marker="x")
        # plt.scatter(percentage_noise_0_dim, ph_dim_1_dim_true, c="r", marker="o")
        # plt.scatter(percentage_noise_0_dim, ph_dim_1_dim_mutated, c="r", marker="x")
        # plt.xticks()
        # plt.yticks()
        # plt.xlabel("Percentage of noise")
        # plt.ylabel("Estimated dim$_{PH}$")
        # plt.legend(["dim$_{PH}^0$ Original", "dim$_{PH}^0$ Perturbed", "dim$_{PH}^1$ Original", "dim$_{PH}^1$ Perturbed"], bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.title(f"PH dimension of {no_data_point} points on a {data_type}\nWith perturbed data by {typeToText(type_noise)}")
        # # plt.show()
        # plt.savefig('./results/Plots/PointCloud/PHDimDataPlots_'+type_noise+'/' + str(data_type) + ".png")
        # plt.clf()


drawFromPertubation("FlipPoints", "AlexNet")