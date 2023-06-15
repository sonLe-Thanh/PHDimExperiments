from TestSets import *
from PHDimPointCloud import *
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np



max_points = 2000
# percentage = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

dimension = [0,1]
alpha = 1
max_sampling_size = 1000
step_size = 50
level = 8
outer_dia = 5
inner_dia = 3
mean = 2
var = 0.2
data_dim = 10

percentage = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]



for percent_noise in percentage:
    # break
    no_noise = int(max_points * percent_noise)
    no_points = max_points - no_noise

    data = sampleSierpinski2D(no_points, level)
    # data = sampleDisk2D(no_points)
    # data = sampleCantorDust2D(no_points, level)
    # data = sampleCantorSetCrossInterval(no_points, level)
    # data = sampleCantorDust3D(no_points, level)
    # data = sampleAnnulus2D(no_points, outer_dia, inner_dia)
    # data = sampleTorus3D(no_points, outer_dia, inner_dia)
    # data = sampleNormal3D(no_points, mean, var)
    # data = sampleNormalND(no_points, data_dim, mean, var)
    # data = sampleDiskND(no_points, data_dim)
    # data = sampleClintonTorus(no_points)
    # data = sampleClintonTorusND(no_points, data_dim)
    # noise = sampleNoise(no_noise, data.shape[1], np.std(data))
    #
    # data_with_noise = np.concatenate((data, noise))

    # for dim in dimension:
    #     log_n, log_alpha_sum, est_dim, LR_fit = estimatePersistentHomologyDimension(data_with_noise, dim, alpha,
    #                                                                             max_sampling_size, step_size)
    #
    #     with open("results/PHDimData_SampleNoise/10DClintonTorus.txt", 'a') as file:
    #         file.write(f"{max_points}, {percent_noise}, {dim}, {est_dim}\n")


    true_data = deepcopy(data)
    # mutated_data, unchanged, mutated = flipPoints(data, percent_noise)
    # mutated_data, unchanged, mutated = perturbedPoints(data, percent_noise, is_random=True)
    # mutated_data, unchanged, mutated = reduceToZero(data, percent_noise)
    mutated_data, unchanged, mutated = inversePoints(data, percent_noise)
    # plt.scatter(unchanged[:,0], unchanged[:, 1], c='b', marker='.')
    # plt.scatter(mutated[:,0], mutated[:, 1], c='r', marker='x')
    # plt.scatter(mutated_data[:,0], mutated_data[:, 1], c='b', marker='o')
    # plt.scatter(true_data[:, 0], true_data[:, 1], c='r', marker='x')
    # plt.show()

    for dim in dimension:
        log_n, log_alpha_sum, est_dim, _ = estimatePersistentHomologyDimension(mutated_data, dim, alpha,
                                                                                max_sampling_size, step_size)
        log_n_real, log_alpha_sum_real, est_dim_real, _ = estimatePersistentHomologyDimension(true_data, dim, alpha,
                                                                               max_sampling_size, step_size)
        with open("results/PHDimReport/PointClouds/PHDimData_InversePoint/2DSierpinski.txt", 'a') as file:
            file.write(f"Mutated: {max_points}, {percent_noise}, {dim}, {est_dim}\nTrue: {max_points}, {percent_noise}, {dim}, {est_dim_real}\n")