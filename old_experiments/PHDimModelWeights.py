from TestSets import *
from PHDimPointCloud import *
from copy import deepcopy
import glob

import numpy as np


dimension = [0, 1]
alpha = 1
max_sampling_size = 1000
step_size = 50

percentage = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

full_files = glob.glob('results/WeightsHistories/*')

for file_name in full_files:
    # only for AlexNet_CIFAR10 atm
    train_info = file_name[:-4].split('/')[-1]
    model_name = train_info.split('|')[0]
    dataset_name = train_info.split('|')[1]
    batch_size = int(train_info.split('|')[2].split(':')[1])
    optim_name = train_info.split('|')[3].split(':')[1]
    learning_rate = float(train_info.split('|')[-1].split(':')[1])

    weight_hist = np.load(file_name)

    max_points = weight_hist.shape[0]


    for dim in dimension:
        log_n_real, log_alpha_sum_real, est_dim_real, _ = estimatePersistentHomologyDimension(weight_hist, dim, alpha,
                                                                                              max_sampling_size,
                                                                                              step_size)
        with open("results/PHDimReport/PHDimModelWeights/FlipPoints/"+model_name+".txt", 'a') as file:
            file.write(
                f"{train_info} True: {max_points}, {0}, {dim}, {est_dim_real}\n")

    for percent_noise in percentage:
        true_data = deepcopy(weight_hist)
        mutated_data, unchanged, mutated = flipPoints(weight_hist, percent_noise)
        # mutated_data, unchanged, mutated = perturbedPoints(weight_hist, percent_noise, is_random=True)
        # mutated_data, unchanged, mutated = reduceToZero(weight_hist, percent_noise)
        # mutated_data, unchanged, mutated = inversePoints(weight_hist, percent_noise)
        for dim in dimension:
            log_n, log_alpha_sum, est_dim, _ = estimatePersistentHomologyDimension(mutated_data, dim, alpha,
                                                                                    max_sampling_size, step_size)
            with open("results/PHDimReport/PHDimModelWeights/FlipPoints/"+model_name+".txt", 'a') as file:
                file.write(f"{train_info} Mutated: {max_points}, {percent_noise}, {dim}, {est_dim}\n")
