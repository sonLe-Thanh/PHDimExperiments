import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
from PHDimDisk import calculateAvgVar


def typeToText(type_noise: str):
    if type_noise == "FlipPoints":
        return "flipping some points"
    elif type_noise == "InversePoint":
        return "inverting some points"
    elif type_noise == "Perturbed":
        return "moving some points by a random value"
    elif type_noise == "Reduced0":
        return "reducing some points to 0"
    else:
        return "perturbing"


def drawFromSampleNoise():
    full_files = glob.glob('results/PHDimReport/PointClouds/PHDimData_SampleNoise/*')
    for full_path in full_files:

        data_type = full_path.split('/')[4].split('.')[0]

        data = genfromtxt(full_path, delimiter=',', dtype=float)
        no_data_point = int(data[0, 0])

        percentage_noise_0_dim = []
        percentage_noise_1_dim = []

        ph_dim_0_dim = []
        ph_dim_1_dim = []

        for i in range(data.shape[0]):
            if int(data[i, 2]) == 0:
                percentage_noise_0_dim.append(data[i, 1])
                ph_dim_0_dim.append(data[i, 3])
            elif int(data[i, 2]) == 1:
                percentage_noise_1_dim.append(data[i, 1])
                ph_dim_1_dim.append(data[i, 3])
            else:
                pass

        plt.plot(percentage_noise_0_dim, ph_dim_0_dim, 'b')
        plt.scatter(percentage_noise_0_dim, ph_dim_0_dim)
        plt.plot(percentage_noise_0_dim, ph_dim_1_dim, 'r')
        plt.scatter(percentage_noise_0_dim, ph_dim_1_dim)
        plt.xticks()
        plt.yticks()
        plt.xlabel("Percentage of noise")
        plt.ylabel("Estimated dim$_{PH}$")
        plt.legend(["dim$_{PH}^0$", "dim$_{PH}^1$"])
        plt.title(f"Persistence homology dimension of {no_data_point} points on a {data_type}")
        plt.savefig('./results/Plots/PointCloud/PHDimDataPlots_SampleNoise/' + str(data_type) + ".png")
        plt.clf()


def drawFromPertubation(type_noise: str):
    full_files = glob.glob('results/PHDimReport/PointClouds/PHDimData_' + type_noise + '/*')
    for full_path in full_files:
        data_type = full_path.split('/')[4].split('.')[0]

        data = genfromtxt(full_path, delimiter=':', dtype=str)
        data = np.concatenate((np.delete(data, 1, -1), np.array(list(map(lambda x: x.split(','), data[:, 1])))), axis=1)

        no_data_point = int(data[0, 1])

        percentage_noise_0_dim = []
        percentage_noise_1_dim = []

        ph_dim_0_dim_true = []
        ph_dim_1_dim_true = []

        ph_dim_0_dim_mutated = []
        ph_dim_1_dim_mutated = []

        for i in range(data.shape[0]):
            if data[i, 0] == "True":
                if int(data[i, 3]) == 0:
                    ph_dim_0_dim_true.append(float(data[i, 4]))
                elif int(data[i, 3]) == 1:
                    ph_dim_1_dim_true.append(float(data[i, 4]))
                else:
                    raise ValueError("Not recognizing value")
            elif data[i, 0] == "Mutated":
                if int(data[i, 3]) == 0:
                    ph_dim_0_dim_mutated.append(float(data[i, 4]))
                    percentage_noise_0_dim.append(float(data[i, 2]))
                elif int(data[i, 3]) == 1:
                    ph_dim_1_dim_mutated.append(float(data[i, 4]))
                    percentage_noise_1_dim.append(float(data[i, 2]))
                else:
                    raise ValueError("Not recognizing value")
            else:
                raise ValueError("Not recognizing value")

        plt.subplots_adjust(right=0.7)
        plt.plot(percentage_noise_0_dim, ph_dim_0_dim_true, 'b')
        plt.plot(percentage_noise_0_dim, ph_dim_0_dim_mutated, '--b')
        plt.plot(percentage_noise_0_dim, ph_dim_1_dim_true, 'r')
        plt.plot(percentage_noise_0_dim, ph_dim_1_dim_mutated, '--r')

        plt.scatter(percentage_noise_0_dim, ph_dim_0_dim_true, c="b", marker="o")
        plt.scatter(percentage_noise_0_dim, ph_dim_0_dim_mutated, c="b", marker="x")
        plt.scatter(percentage_noise_0_dim, ph_dim_1_dim_true, c="r", marker="o")
        plt.scatter(percentage_noise_0_dim, ph_dim_1_dim_mutated, c="r", marker="x")
        plt.xticks()
        plt.yticks()
        plt.xlabel("Percentage of noise")
        plt.ylabel("Estimated dim$_{PH}$")
        plt.legend(
            ["dim$_{PH}^0$ Original", "dim$_{PH}^0$ Perturbed", "dim$_{PH}^1$ Original", "dim$_{PH}^1$ Perturbed"],
            bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(
            f"PH dimension of {no_data_point} points on a {data_type}\nWith perturbed data by {typeToText(type_noise)}")
        # plt.show()
        plt.savefig('./results/Plots/PointCloud/PHDimDataPlots_' + type_noise + '/' + str(data_type) + ".png")
        plt.clf()


def drawFromDataWRTNoise(file_name, percent_noise, save_path):
    full_path = glob.glob('results/PHDimReport/PointClouds/' + file_name)
    data = genfromtxt(full_path[0], delimiter=',', dtype=str)

    # 0: no points, 1:% Noise, 2: % eps/R, 3: PH Dim, 4: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    ratio = []
    for i in range(data.shape[0]):
        # Percent noise
        if float(data[i, 1]) == percent_noise:
            if int(data[i, 3]) == 0:
                ratio.append(float(data[i, 2]))
                est_PH_Dim_0.append(float(data[i, 4]))
            else:
                est_PH_Dim_1.append(float(data[i, 4]))

    plt.subplots_adjust(right=0.7)
    plt.plot(ratio, est_PH_Dim_0, 'b')
    plt.plot(ratio, est_PH_Dim_1, 'r')

    plt.scatter(ratio, est_PH_Dim_0, c="b", marker="o")
    plt.scatter(ratio, est_PH_Dim_1, c="b", marker="x")

    plt.xticks()
    plt.yticks()
    plt.xlabel("$eps/R$")
    plt.ylabel("Estimated dim$_{PH}$")
    plt.legend(["dim$_{PH}^0$", "dim$_{PH}^1$"], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(
        f"PH dimension of {data[0, 0]} points on a 2D Disk\nWith {percent_noise * 100}% perturbed data by adding normal noise")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/Noise{percent_noise}.png')
    plt.clf()


def drawFromDataWRTRatio(file_name, ratio, save_path):
    full_path = glob.glob('results/PHDimReport/PointClouds/' + file_name)
    data = genfromtxt(full_path[0], delimiter=',', dtype=str)

    # 0: no points, 1:% Noise, 2: % eps/R, 3: PH Dim, 4: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    percent_noise = []
    for i in range(data.shape[0]):
        # Percent noise
        if float(data[i, 2]) == ratio:
            if int(data[i, 3]) == 0:
                percent_noise.append(float(data[i, 1]) * 100)
                est_PH_Dim_0.append(float(data[i, 4]))
            else:
                est_PH_Dim_1.append(float(data[i, 4]))

    plt.subplots_adjust(right=0.7)
    plt.plot(percent_noise, est_PH_Dim_0, 'b')
    plt.plot(percent_noise, est_PH_Dim_1, 'r')

    plt.scatter(percent_noise, est_PH_Dim_0, c="b", marker="o")
    plt.scatter(percent_noise, est_PH_Dim_1, c="b", marker="x")

    plt.xticks()
    plt.yticks()
    plt.xlabel("Percentage of noise")
    plt.ylabel("Estimated dim$_{PH}$")
    plt.legend(["dim$_{PH}^0$", "dim$_{PH}^1$"], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(
        f"PH dimension of {data[0, 0]} points on a 3D Disk\nWith perturbed data by adding normal noise of ratio {ratio}")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/Ratio{ratio}.png')
    plt.clf()


def drawFromDataWRTNoiseError(data, error, percent_noise, save_path):
    """
    :param data: np array
    :param error: np array
    :param percent_noise: float
    :param save_path: str
    """
    # 0: no points, 1:% Noise, 2: % eps/R, 3: PH Dim, 4: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    PH_Dim_0_err = []
    PH_Dim_1_err = []
    ratio = []
    for i in range(data.shape[0]):
        # Percent noise
        if float(data[i, 1]) == percent_noise:
            if int(data[i, 3]) == 0:
                ratio.append(float(data[i, 2]))
                est_PH_Dim_0.append(float(data[i, 4]))
                PH_Dim_0_err.append(float(error[i, 4]))
            else:
                est_PH_Dim_1.append(float(data[i, 4]))
                PH_Dim_1_err.append(float(error[i, 4]))

    plt.subplots_adjust(right=0.7)
    plt.errorbar(ratio, est_PH_Dim_0, yerr=PH_Dim_0_err, fmt='bo-', ecolor='royalblue', elinewidth=0.8, capsize=3)
    plt.errorbar(ratio, est_PH_Dim_1, yerr=PH_Dim_1_err, fmt='ro-', ecolor='deeppink', elinewidth=0.8, capsize=3)

    plt.xticks()
    plt.yticks()
    plt.xlabel("$eps/R$")
    plt.ylabel("Estimated dim$_{PH}$")
    plt.legend(["dim$_{PH}^0$", "dim$_{PH}^1$"], bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.title(
        f"PH dimension of {round(data[0, 0], 1)} points on a 3D Disk\nWith {percent_noise * 100}% perturbed data by adding normal noise")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/NoiseWError{percent_noise}.png')
    plt.clf()


def drawFromDataWRTNoise3D(data, percent_noise, save_path):
    """
    :param data: np array, from PHDimDisk
    :param percent_noise: float
    :param save_path: str
    """
    # 0: no points, 1: radius, 2:% Noise, 3: % eps/R, 4: PH Dim, 5: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    ratio = []
    radius = []
    for i in range(data.shape[0]):
        # Percent noise
        if float(data[i, 2]) == percent_noise:
            if int(data[i, 4]) == 0:
                ratio.append(float(data[i, 3]))
                radius.append(float(data[i, 1]))

                est_PH_Dim_0.append(float(data[i, 5]))
            else:
                est_PH_Dim_1.append(float(data[i, 5]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plt.subplots_adjust(right=0.8)
    ax.plot3D(radius, ratio, est_PH_Dim_0, label="dim$_{PH}^0$", marker="o", c="b")
    ax.plot3D(radius, ratio, est_PH_Dim_1, label="dim$_{PH}^1$", marker="x", c="r")
    ax.legend()
    ax.set_xlabel("Radius")
    ax.set_ylabel("$eps/R$")
    ax.set_zlabel("Estimated dim$_{PH}$")

    ax.set_title(
        f"PH dimension of {round(data[0, 0], 0)} points on a 10D Disk\nWith {percent_noise * 100}% perturbed data by adding normal noise")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/NoiseWError{percent_noise}.png')
    plt.clf()


def drawFromDataWRTRatioError(data, error, ratio, save_path):
    """
    :param data: np array
    :param error: np array
    :param ratio: float
    :param save_path: str
    """
    # 0: no points, 1:% Noise, 2: % eps/R, 3: PH Dim, 4: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    percent_noise = []
    PH_Dim_0_err = []
    PH_Dim_1_err = []
    for i in range(data.shape[0]):
        # Percent noise
        if float(data[i, 2]) == ratio:
            if int(data[i, 3]) == 0:
                percent_noise.append(float(data[i, 1]) * 100)
                est_PH_Dim_0.append(float(data[i, 4]))
                PH_Dim_0_err.append(float(error[i, 4]))
            else:
                est_PH_Dim_1.append(float(data[i, 4]))
                PH_Dim_1_err.append(float(error[i, 4]))

    plt.subplots_adjust(right=0.7)
    plt.errorbar(percent_noise, est_PH_Dim_0, yerr=PH_Dim_0_err, fmt='bo-', ecolor='royalblue', elinewidth=0.8,
                 capsize=3)
    plt.errorbar(percent_noise, est_PH_Dim_1, yerr=PH_Dim_1_err, fmt='ro-', ecolor='deeppink', elinewidth=0.8,
                 capsize=3)

    plt.xticks()
    plt.yticks()
    plt.xlabel("Percentage of noise")
    plt.ylabel("Estimated dim$_{PH}$")
    plt.legend(["dim$_{PH}^0$", "dim$_{PH}^1$"], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(
        f"PH dimension of {data[0, 0]} points on a 3D Disk\nWith perturbed data by adding normal noise of ratio {round(ratio, 1)}")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/RatioWError{ratio}.png')
    plt.clf()
    plt.close()

def drawFromDataWRTRatio3D(data, ratio, save_path):
    """
    :param data: np array, from PHDimDisk
    :param ratio: float
    :param save_path: str
    """
    # 0: no points, 1: radius, 2:% Noise, 3: % eps/R, 4: PH Dim, 5: est PH Dim
    est_PH_Dim_0 = []
    est_PH_Dim_1 = []
    percent_noise = []
    radius = []
    for i in range(data.shape[0]):
        # ratio
        if float(data[i, 3]) == ratio:
            if int(data[i, 4]) == 0:
                percent_noise.append(float(data[i, 2]))
                radius.append(float(data[i, 1]))

                est_PH_Dim_0.append(float(data[i, 5]))
            else:
                est_PH_Dim_1.append(float(data[i, 5]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plt.subplots_adjust(right=0.8)
    ax.plot3D(radius, percent_noise, est_PH_Dim_0, label="dim$_{PH}^0$", marker="o", c="b")
    ax.plot3D(radius, percent_noise, est_PH_Dim_1, label="dim$_{PH}^1$", marker="x", c="r")
    ax.legend()
    ax.set_xlabel("Radius")
    ax.set_ylabel("Percentage of noise")
    ax.set_zlabel("Estimated dim$_{PH}$")

    ax.set_title(
        f"PH dimension of {round(data[0, 0], 0)} points on a 10D Disk\nWith perturbed data by adding normal noise of "
        f"ratio {round(ratio, 1)}")
    # plt.show()
    plt.savefig(f'./results/Plots/PointCloud/{save_path}/RatioWError{ratio}.png')
    plt.clf()
    plt.close()


def drawFromDataMultipleDim3D(data_2D, data_5D, data_10D, save_path):
    """
    :param data_2D, data_5D, data_10D: np arrays, from PHDimDisk
    :param percent_noise: float
    :param save_path: str
    """

    # 0: no points, 1: radius, 2:% Noise, 3: % eps/R, 4: PH Dim, 5: est PH Dim
    est_PH_Dim_0_2D = []
    est_PH_Dim_1_2D = []
    est_PH_Dim_0_5D = []
    est_PH_Dim_1_5D = []
    est_PH_Dim_0_10D = []
    est_PH_Dim_1_10D = []
    ratio = []
    percent_noise = []
    # Only need to consider the case when r = 1
    for i in range(data_2D.shape[0]):
        # Percent noise
        if float(data_2D[i, 1]) == 1:
            if int(data_2D[i, 4]) == 0:
                percent_noise.append(data_2D[i, 2])
                ratio.append(float(data_2D[i, 3]))

                est_PH_Dim_0_2D.append(float(data_2D[i, 5]))
                est_PH_Dim_0_5D.append(float(data_5D[i, 5]))
                est_PH_Dim_0_10D.append(float(data_10D[i, 5]))
            else:
                est_PH_Dim_1_2D.append(float(data_2D[i, 5]))
                est_PH_Dim_1_5D.append(float(data_5D[i, 5]))
                est_PH_Dim_1_10D.append(float(data_10D[i, 5]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plt.subplots_adjust(right=0.8)
    # ax.plot3D(percent_noise, ratio, est_PH_Dim_0_2D, label="2D Disk", marker="o", c="b")
    ax.plot3D(percent_noise, ratio, est_PH_Dim_1_2D, label="2D Disk$", marker="x", c="b")

    # ax.plot3D(percent_noise, ratio, est_PH_Dim_0_5D, label="5D Disk", marker="o", c="r")
    ax.plot3D(percent_noise, ratio, est_PH_Dim_1_5D, label="5D Disk$", marker="x", c="r")

    # ax.plot3D(percent_noise, ratio, est_PH_Dim_0_10D, label="10D Disk", marker="o", c="g")
    ax.plot3D(percent_noise, ratio, est_PH_Dim_1_10D, label="10D Disk$", marker="x", c="g")

    # fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    ax.set_xlabel("Percentage of noise")
    ax.set_ylabel("$eps/R$")
    ax.set_zlabel("Estimated dim$_{PH}$")

    ax.set_title(
        f"PH$_1$ dimension of {int(data_2D[0, 0])} points on a disk")
    # plt.show()
    plt.savefig(save_path)
    plt.clf()


######### Draw wrt to noise and ratio
# Parameters for Radius = 1 :
# percent_noise_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
# percent_noise_list = [0.1, 0.15, 0.2]
# radius = 1
# eps_list = np.arange(1, 2 * radius, 0.2)
# eps_list /= radius

# Parameters for Radius = 2.5
# percent_noise_list = [0.1, 0.15, 0.2]
# radius = 2.5
# eps_list = np.arange(1, 2 * radius, 0.2)
# eps_list /= radius
#
# # Parameters for Radius = 5
# # percent_noise_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
# # eps_list = [0.1, 0.5, 0.8, 1, 1.2, 1.4, 1.5, 1.8, 2]
#
#
# avg, error = calculateAvgVar('results/PHDimReport/PointClouds/PHDim3DDisk/Radius2.5/')
#
# for percent_noise in percent_noise_list:
#     drawFromDataWRTNoiseError(avg, error, percent_noise, '3DDiskPerturbed/AvgWError_Radius2.5')
#
# for eps in eps_list:
#     drawFromDataWRTRatioError(avg, error, eps, '3DDiskPerturbed/AvgWError_Radius2.5')


############ Draw for different radii

# percent_noise_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
# eps_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
#
# avg, error = calculateAvgVar('results/PHDimReport/PointClouds/PHDimDisk/2D/')
# for percent_noise in percent_noise_list:
#     drawFromDataWRTNoise3D(avg, percent_noise, "2DDiskPerturbed/DifferentRadii")
#
# for eps in eps_list:
#     drawFromDataWRTRatio3D(avg, eps, "2DDiskPerturbed/DifferentRadii")


avg_2D, _ = calculateAvgVar('results/PHDimReport/PointClouds/PHDimDisk/2D/')
avg_5D, _ = calculateAvgVar('results/PHDimReport/PointClouds/PHDimDisk/5D/')
avg_10D, _ = calculateAvgVar('results/PHDimReport/PointClouds/PHDimDisk/10D/')
drawFromDataMultipleDim3D(avg_2D, avg_5D, avg_10D, './results/Plots/PointCloud/Disk/PHDim1.png')