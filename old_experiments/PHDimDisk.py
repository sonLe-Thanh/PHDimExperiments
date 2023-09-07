# Test PH Dim with noise

from TestSets import *
from copy import deepcopy
import glob
from numpy import genfromtxt
from functools import reduce

import numpy as np


def perturbedDisk(data, percentage, eps):
    no_samples = int(data.shape[0] * percentage)
    mutated_idx = np.random.choice(data.shape[0], no_samples, replace=False)
    noisy_data = deepcopy(data)
    for idx in mutated_idx:
        noisy_data[idx] += np.random.normal(0.0, eps, (1, noisy_data.shape[1]))[0]

    return noisy_data, noisy_data[mutated_idx], noisy_data[~np.isin(np.arange(data.shape[0]), mutated_idx)]


def runExperiment(type_data:str, num_run:str):
    dimension = 1
    alpha = 1
    max_sampling_size = 1000
    step_size = 50
    max_points = 1000
    radius = 2.5

    percent_noise_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    # eps_list = [0.1, 0.5, 0.8, 1, 1.2, 1.4, 1.5, 1.8, 2]
    eps_list = np.arange(1, 2 * radius, 0.2)
    eps_list /= radius

    if type_data == "2DDisk":
        disk = sampleAnnulus2D(max_points, radius, radius + 0.5)
    elif "3DDisk" in type_data:
        disk = sampleSphere(max_points, radius, int(type_data[0]))
    elif "10DDisk" in type_data:
        disk = sampleSphere(max_points, radius, 10)

    # run1-5: radius = 5, eps list, run6-10: radius = 1 esp = eps_list /= radius, run11-15: radius=0.5, eps_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for percent_noise in percent_noise_list:
        for eps_over_r in eps_list:
            noisy_disk, mutated_disk, unmutated_disk = perturbedDisk(disk, percent_noise, eps_over_r)
            PHDim_list = estimateMultiplePersistentHomologyDimension(noisy_disk, dimension, alpha, max_sampling_size,
                                                                     step_size)
            with open("results/PHDimReport/PointClouds/PHDim"+type_data+"/"+num_run+".txt", 'a') as file:
                for dim in range(len(PHDim_list)):
                    file.write(f"{max_points}, {percent_noise}, {eps_over_r}, {dim}, {PHDim_list[dim]}\n")



def runExperimentDiskDiffRadius(num_run:str, radius_list):
    dimension = 1
    alpha = 1
    max_sampling_size = 1000
    step_size = 50
    max_points = 1000

    percent_noise_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    # percent_noise_list = [0.1, 0.15, 0.2]
    # eps_list = [0.1, 0.5, 0.8, 1, 1.2, 1.4, 1.5, 1.8, 2]
    eps_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    for radius in radius_list:
        disk = sampleSphere(max_points, radius, 5)
        for percent_noise in percent_noise_list:
            for eps in eps_list:
                noisy_disk, mutated_disk, unmutated_disk = perturbedDisk(disk, percent_noise, eps)
                PHDim_list = estimateMultiplePersistentHomologyDimension(noisy_disk, dimension, alpha, max_sampling_size,                                                                             step_size)
                with open("results/PHDimReport/PointClouds/PHDimDisk/10D/"+num_run+".txt", 'a') as file:
                    for dim in range(len(PHDim_list)):
                        file.write(f"{max_points}, {radius}, {percent_noise}, {eps}, {dim}, {PHDim_list[dim]}\n")


def calculateAvgVar(folder_path):
    full_files = glob.glob(folder_path + '*')
    datas = [genfromtxt(file, delimiter=',', dtype=float) for file in full_files if "NotTaken" not in file]

    avg_data = reduce(lambda x, y: x + y, datas) / len(datas)
    error = reduce(lambda x, y: np.maximum(x, y), [data - avg_data for data in datas])

    return avg_data, error


# run_list = ["run1"]
# radii = [1, 2.5, 3.75, 5]
# for run in run_list:
#     runExperimentDiskDiffRadius(run, radii)