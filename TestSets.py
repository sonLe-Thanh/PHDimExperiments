import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin, hypot
import itertools
import random
import copy

def sampleCantorSet(no_points, level):
    points = np.zeros((1, no_points))
    bin_sequence = np.random.randint(low=0, high=2, size=[no_points, level])
    for i in range(level):
        points += 2 * (bin_sequence[:, i] / (3 ** i))
    return points.T


def sampleCantorDust2D(no_points, level):
    x_points = sampleCantorSet(no_points, level)
    y_points = sampleCantorSet(no_points, level)
    return np.concatenate((x_points, y_points), axis=1)


def sampleCantorDust3D(no_points, level):
    x_points = sampleCantorSet(no_points, level)
    y_points = sampleCantorSet(no_points, level)
    z_points = sampleCantorSet(no_points, level)
    return np.concatenate((x_points, y_points, z_points), axis=1)


def sampleCantorSetCrossInterval(no_points, level):
    x_points = sampleCantorSet(no_points, level)
    y_points = np.random.rand(no_points, 1)
    return np.concatenate((x_points, y_points), axis=1)


def sampleDisk2D(no_points):
    points = np.zeros((1, 2))
    i = 0
    while i < no_points:
        new_point = 2 * np.random.rand(1, 2) / sqrt(pi) - 1 / sqrt(pi)
        if np.linalg.norm(new_point, 2) <= 1 / sqrt(pi):
            points = np.concatenate((points, new_point), axis=0)
            i += 1
    return np.array(points[1:, :])


def sampleTriangle2D(no_points):
    points = np.zeros((1, 2))
    i = 0
    while i < no_points:
        new_point = 2 * np.random.rand(1, 2) * (3 ** (-1 / 4)) - np.array([3 ** (-1 / 4), 0])
        if new_point[:, 1] <= -sqrt(3) * new_point[:, 0] + 3 ** (1 / 4) and new_point[:, 1] <= sqrt(3) * new_point[:,
                                                                                                         0] + 3 ** (
                1 / 4):
            points = np.concatenate((points, new_point), axis=0)
            i += 1
    return np.array(points[1:, :])


def sampleSierpinski2D(no_points, level, separation=0):
    ten_sequence = np.random.randint(low=0, high=3, size=[no_points, level])
    points = np.zeros((no_points, 2))
    for i in range(1, level + 1):
        first_part = second_part = np.zeros((no_points, 2))
        first_part[ten_sequence[:, i - 1] == 1] = np.array([1, 0]) / ((2 + separation) ** (i - 1))
        second_part[ten_sequence[:, i - 1] == 2] = np.array([1 / 2, sqrt(3) / 2]) / ((2 + separation) ** (i - 1))
        points += first_part + second_part
    return points


def samplePointAnnulus2D(outer_dia, inner_dia, x_inner, y_inner):
    """
        Sample uniformly from (x, y) satisfiying:
            x**2 + y**2 <= r_outer**2
            (x-x_inner)**2 + (y-y_inner)**2 > r_inner**2
        Assumes that the inner circle lies inside the outer circle;
        i.e., that hypot(x_inner, y_inner) <= r_outer - r_inner.
    """
    # Sample from a normal annulus with radii r_inner and r_outer.
    rad = sqrt(np.random.uniform(inner_dia ** 2, outer_dia ** 2))
    angle = np.random.uniform(-pi, pi)
    x, y = rad * cos(angle), rad * sin(angle)

    # If we're inside the forbidden hole, reflect.
    if hypot(x - x_inner, y - y_inner) < inner_dia:
        x, y = x_inner - x, y_inner - y

    return [x, y]


def sampleAnnulus2D(no_points, outer_dia, inner_dia):
    annulus = []
    for _ in range(no_points):
        annulus.append(samplePointAnnulus2D(outer_dia, inner_dia, 0, 0))

    return np.array(annulus)


def sampleTorus3D(no_points, outer_dia, inner_dia):
    # Sample from a normal annulus with radii r_inner and r_outer.
    theta1 = 2 * np.pi * np.random.rand(no_points)
    theta2 = 2 * np.pi * np.random.rand(no_points)

    x = (outer_dia + inner_dia * np.cos(theta2)) * np.cos(theta1)
    y = (outer_dia + inner_dia * np.cos(theta2)) * np.sin(theta1)
    z = inner_dia * np.sin(theta2)

    return np.array([x, y, z]).T


def sampleNoise(no_samples, no_dim, scale):
    return scale * np.random.randn(no_samples, no_dim)


def sampleNormal3D(no_samples, mean, var):
    return mean * np.random.rand(no_samples, 3) + var


def sampleNormalND(no_samples, data_dim, mean, var):
    return mean * np.random.rand(no_samples, data_dim) + var


def sampleDiskND(no_points, data_dim):
    points = np.zeros((1, data_dim))
    i = 0
    while i < no_points:
        new_point = 2 * np.random.rand(1, data_dim) / sqrt(pi) - 1 / sqrt(pi)
        if np.linalg.norm(new_point, 2) <= 1 / sqrt(pi):
            points = np.concatenate((points, new_point), axis=0)
            i += 1
    return np.array(points[1:, :])


def sampleSphere(no_points, radius=1.0, data_dim=3):
    data = np.random.randn(data_dim, no_points)
    data /= np.linalg.norm(data, axis=0)
    data *= radius
    return data.T

def sampleClintonTorus(no_points):
    theta1 = 2 * np.pi * np.random.rand(no_points)
    theta2 = 2 * np.pi * np.random.rand(no_points)

    cord_1, cord_2, cord_3, cord_4 = np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)

    return np.array([cord_1, cord_2, cord_3, cord_4]).T / sqrt(2)


def sampleClintonTorusND(no_points, dim_data):
    no_var = int(dim_data / 2)
    coeff = 1 / sqrt(dim_data)
    theta = []
    cord_1 = []
    cord_2 = []
    for _ in range(no_var):
        theta.append(2 * np.pi * np.random.rand(no_points))

    for i in range(no_var):
        cord_1.append(np.sin(theta[i]))
        cord_2.append(np.cos(theta[i]))
    cord_1 = np.array(cord_1).T
    cord_2 = np.array(cord_2).T
    res = np.zeros((cord_1.shape[0], 1))
    for i in range(cord_1.shape[1]):
        if i > 0:
            res = np.vstack((res, cord_1[:, i], cord_2[:, i]))
        else:
            res = np.vstack((cord_1[:, i], cord_2[:, i]))

    return res.T * coeff


def flipPoints(data, percentage):
    no_samples = int(data.shape[0] * percentage)
    mutated_idx = np.random.choice(data.shape[0], no_samples, replace=False)
    data[mutated_idx] = - data[mutated_idx]
    return data, data[mutated_idx], data[~np.isin(np.arange(data.shape[0]), mutated_idx)]

def movingPoint(vector, max_taken=1000):
    taken = []
    len = vector.shape[0]
    i = 0
    while i < max_taken:
        idx = random.randint(0, len - 1)
        if idx not in taken:
            taken.append(idx)
            perturbed_vector = copy.deepcopy(vector)
            perturbed_vector[idx] += 0.01


def perturbedPoints(data, percentage, discrepancy=0.1, is_random=False):
    no_samples = int(data.shape[0] * percentage)
    mutated_idx = np.random.choice(data.shape[0], no_samples, replace=False)

    if is_random:
        data[mutated_idx] += np.random.uniform(low=-1, high=1, size=(no_samples, 1))
    else:
        data[mutated_idx] += discrepancy

    return data, data[mutated_idx], data[~np.isin(np.arange(data.shape[0]), mutated_idx)]


def reduceToZero(data, percentage):
    no_samples = int(data.shape[0] * percentage)
    mutated_idx = np.random.choice(data.shape[0], no_samples, replace=False)
    data[mutated_idx] = 0
    return data, data[mutated_idx], data[~np.isin(np.arange(data.shape[0]), mutated_idx)]


def inversePoints(data, percentage):
    no_samples = int(data.shape[0] * percentage)
    mutated_idx = np.random.choice(data.shape[0], no_samples, replace=False)
    data[mutated_idx] = 1 / data[mutated_idx]
    return data, data[mutated_idx], data[~np.isin(np.arange(data.shape[0]), mutated_idx)]


def sampleDiskND(no_points, radius=1.0, no_dim=3):
    points = np.zeros((1, no_dim))
    i = 0
    while i < no_points:
        new_point = 2 * np.random.rand(1, no_dim) / sqrt(pi) - 1 / sqrt(pi)
        if np.linalg.norm(new_point, 2) <= 1 / sqrt(pi):
            points = np.concatenate((points, new_point), axis=0)
            i += 1
    return np.array(points[1:, :])

# no_points = 10000
# level = 6
# sep = 0
# cantor_dust = sampleSierpinski2D(no_points, level)
#
# plt.scatter(cantor_dust[:, 0], cantor_dust[:, 1], c='r', marker='.')
# plt.show()
