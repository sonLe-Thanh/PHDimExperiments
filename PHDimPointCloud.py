from TestSets import *

import numpy as np
import gudhi as gd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from ripser import ripser


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
    # Make the inf value equals the max value if there are inf, it is guaranteed that there is only one such element
    if np.isposinf(barcodes[-1][1]):
        barcodes[-1][1] = barcodes[-2][1]
    # Get the sum
    return (abs(barcodes[:, 1] - barcodes[:, 0]) ** alpha).sum()


def computePersistenceHomology(data, max_dimen=1):
    """
    Compute persistence homology
    Input:
        data: array [no samples, no features]
        max_dimen: maximum dimension to calculate persistence homology
    """
    diagrams = ripser(data, max_dimen)
    return diagrams['dgms']


def estimatePersistentHomologyDimension(data, dimension, alpha, max_sampling_size=1000, no_steps=50):
    """
    Estimate the PH dim of a given data
    Input:
        data: array [no samples, no features]
        dimension: dimension to calculate persistence homology dimension
        alpha: weight for calculate the weighted persistence
        max_sampling_size: maximum sampling size from the data
        no_steps: number of steps for each step of sampling
    """

    # Array to store values
    log_n = []
    log_alpha_sum = []

    no_samples = 100
    while no_samples <= max_sampling_size:
        # Start sample
        samples = data[np.random.choice(data.shape[0], no_samples, replace=False)]
        if dimension == 0:
            dgms = computePersistenceHomology(samples, 1)
        else:
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


def estimateMultiplePersistentHomologyDimension(data, max_dimension, alpha, max_sampling_size=1000, no_steps=50):
    """
    Estimate the PH dim of a given data
    Input:
        data: array [no samples, no features]
        max_dimension: max_dimension to calculate persistence homology dimension, starting from 0
        alpha: weight for calculate the weighted persistence
        max_sampling_size: maximum sampling size from the data
        no_steps: number of steps for each step of sampling
    """

    # Array to store values
    est_PH_dim_list = []

    for dim in range(max_dimension+1):
        _, _, est_PH_dim, _ = estimatePersistentHomologyDimension(data, dim, alpha, max_sampling_size, no_steps)
        est_PH_dim_list.append(est_PH_dim)
    return est_PH_dim_list


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
