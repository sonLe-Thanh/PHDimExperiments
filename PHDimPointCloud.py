from TestSets import *

import numpy as np
import gudhi as gd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from ripser import ripser
from functools import reduce

# Replace ripser to ripder parallel for a bit more speed
from gph import ripser_parallel

def computeTopologyDescriptors(data, max_dimen=1, alpha=1.0):
    """
    :param data:
    :param max_dimen:
    :param alpha:
    :return: E^alpha_0(data), E^alpha_1(data), entropy_0, entropy_1, entropy_sum, dim_PH
    """
    persistence_barcodes = computePersistenceHomology(data, max_dimen)

    # Get the barcodes of dim
    if max_dimen >= len(persistence_barcodes):
        raise ValueError("Dimension out of range")
    # Currently force to take max_dimen = 1
    if max_dimen == 0:
        raise ValueError("Not support this calculation atm")
    # Consider the edge case where there is no barcode
    if len(persistence_barcodes[0]) == 0 or len(persistence_barcodes[1]) == 0:
        return 0, 0, 0, 0, 0, 0

    # Make the inf value equals the max value + 1 if there is inf,
    # if np.isposinf(persistence_barcodes[0][-1][1]):
    #     persistence_barcodes[0][-1][1] = persistence_barcodes[0][-2][1] + 1



    persistence_barcodes[0][np.isinf(persistence_barcodes[0])] = np.nan
    persistence_barcodes[0][np.isnan(persistence_barcodes[0])] = np.max(np.nanmax(persistence_barcodes[0], axis=0) + 1)

    persistence_barcodes[1][np.isinf(persistence_barcodes[1])] = np.nan
    persistence_barcodes[1][np.isnan(persistence_barcodes[1])] = np.max(np.nanmax(persistence_barcodes[1], axis=0) + 1)



    # Calculate total lifetimes
    lifetime_0 = abs(persistence_barcodes[0][:, 1] - persistence_barcodes[0][:, 0])
    lifetime_1 = abs(persistence_barcodes[1][:, 1] - persistence_barcodes[1][:, 0])
    lifetime_total = np.concatenate((lifetime_0, lifetime_1), axis=0)

    # Get the alpha-weighted sums
    e_0 = (lifetime_0 ** alpha).sum()
    e_1 = (lifetime_1 ** alpha).sum()

    # Get the sums for entropy calculation
    total_lifetime_0 = lifetime_0.sum()
    
    total_lifetime_1 = lifetime_1.sum()
    total_lifetime = total_lifetime_0 + total_lifetime_1

    # Calculate entropies
    entropy_0 = - reduce(lambda x, y: x + y / total_lifetime_0 * np.log(y / total_lifetime_0), np.asarray(lifetime_0), 0)
    entropy_1 = - reduce(lambda x, y: x + y / total_lifetime_1 * np.log(y / total_lifetime_1), np.asarray(lifetime_1), 0)
    entropy_total = - reduce(lambda x, y: x + y / total_lifetime * np.log(y / total_lifetime), np.asarray(lifetime_total), 0)

    # Estimate PH_dim^0 (for faster computation)
    max_sampling_size = 2000 if data.shape[0] >= 2000 else data.shape[0]
    no_samples = int(max_sampling_size/2) if data.shape[0] <= 100 else int(max_sampling_size/4)
    no_steps = int((max_sampling_size - no_samples)/5)

    ph_dim_list = []
    # Calculate 10 times for more stability
    no_calculations = 3
    for _ in range(no_calculations):
        _, _, ph_dim, _ = estimatePersistentHomologyDimension(data, 0, 1.0, max_sampling_size=max_sampling_size, no_steps=no_steps, no_samples=no_samples)
        ph_dim_list.append(ph_dim)

    return e_0, e_1, entropy_0, entropy_1, entropy_total, (np.mean(ph_dim_list), np.std(ph_dim_list))

def estimatePersistentHomologyDimension(data, dimension, alpha, max_sampling_size=1000, no_steps=50, no_samples = 100):
    """
    Estimate the PH dim of a given data
    Input:
        data: array [no samples, no features]
        dimension: dimension to calculate persistence homology dimension
        alpha: weight for calculate the weighted persistence
        max_sampling_size: maximum sampling size from the data
        no_steps: number of steps for each step of sampling
        no_samples: number of samples for the starting step
    """

    # Array to store values
    log_n = []
    log_alpha_sum = []

    while no_samples <= max_sampling_size:
        # Start sample
        samples = data[np.random.choice(data.shape[0], no_samples, replace=False)]
        dgms = computePersistenceHomology(samples, dimension)
        weighted_sum = computeAlphaWeightedPersistence(dgms, dimension, alpha)
        log_n.append(np.log(no_samples))
        log_alpha_sum.append(np.log(weighted_sum))

        no_samples += no_steps

    log_n = np.array(log_n).reshape(-1, 1)
    log_alpha_sum = np.array(log_alpha_sum).reshape(-1, 1)

    # LR_fit = LinearRegression().fit(log_n, log_alpha_sum)
    # slope = LR_fit.coef_
    N = len(log_n)
    m = (N * (log_n * log_alpha_sum).sum() - log_n.sum() * log_alpha_sum.sum()) / (
            N * (log_n ** 2).sum() - log_n.sum() ** 2)
    b = log_alpha_sum.mean() - m * log_n.mean()
    # est_dim = alpha / (1 - slope[0][0])
    est_dim = alpha / (1 - m)
    return log_n, log_alpha_sum, est_dim, (m, b)


# compute the alpha weighted sum
def computeAlphaWeightedPersistence(persistence_barcodes, dimension, alpha):
    """
    Compute the alpha weighted persistence of "dimension"
    Input:
        persistence_barcodes: a dim x no_barcodes array of persistence barcodes (results of Ripser), sorted in increasing order
        dimension (int): dimension we want to calculate, starting from 0
        alpha (float)
    Output:
        float: alpha weighted persistence of "dimension"
    """
    # Get the barcodes of dim
    if dimension >= len(persistence_barcodes):
        raise ValueError("Dimension out of range")
    barcodes = persistence_barcodes[dimension]
    # Consider the edge case where there is no barcode
    if len(barcodes) == 0:
        return 0
    # Make the inf value equals the max value + 1 if there are inf, it is guaranteed that there is only one such element
    # if np.isposinf(barcodes[-1][1]):
    #     barcodes[-1][1] = barcodes[-2][1] + 1

    barcodes[np.isinf(barcodes)] = np.nan
    barcodes[np.isnan(barcodes)] = np.max(np.nanmax(persistence_barcodes[0], axis=1) + 1)

    # Get the sum
    return (abs(barcodes[:, 1] - barcodes[:, 0]) ** alpha).sum()


def computePersistenceHomology(data, max_dimen=1, no_threads = 4):
    """
    Compute persistence homology
    Input:
        data: array [no samples, no features]
        max_dimen: maximum dimension to calculate persistence homology
        no_threads: number of threads to execute ripser
    """
    diagrams = ripser_parallel(data, max_dimen, no_threads)
    return diagrams['dgms']

def plotting(dimension, alpha, log_n, log_alpha_sum, estimated_dimension, LR_fitted, type_data):
    if type(LR_fitted) == list:
        log_alpha_sum_pred = LR_fitted[0] * log_n + LR_fitted[1]
    else:
        log_alpha_sum_pred = LR_fitted.predict(log_n)
    plt.scatter(log_n, log_alpha_sum, color="blue", marker="o")
    plt.plot(log_n, log_alpha_sum_pred, color="red", linewidth=3)
    plt.legend(["Predicted data", "Calculated data"], loc="best")
    plt.xlabel("log(n)")
    plt.ylabel(f"log(E^{alpha}_{dimension}(PH))")
    plt.title(
        f"Predicted dimension {estimated_dimension} with the slope of {(estimated_dimension - 1) / estimated_dimension}\n of a " + type_data)
