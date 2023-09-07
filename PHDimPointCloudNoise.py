from TestSets import *
from PHDimPointCloud import *
from copy import deepcopy

max_points = 2000


dimension = [0,1]
alpha = 1
max_sampling_size = 2000
no_samples = 1000
step_size = 100


level = 8
outer_dia = 5
inner_dia = 3
mean = 2
var = 0.2
data_dim = 10

no_repeat = 10

percentage = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

data = sampleSierpinski2D(max_points, level)
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


for dim in dimension:
    est_dim_lst = []
    for _ in range(no_repeat):
        _, _, est_dim, _ = estimatePersistentHomologyDimension(data, dim, alpha, max_sampling_size, step_size, no_samples)
        est_dim_lst.append(est_dim)
    with open("results/PHDimReport/PHDimPointNoise/Sierpinski2D.txt", 'a') as file:
        file.write(f"{max_points}, 0, {dim}, {np.mean(est_dim_lst)}, {np.std(est_dim_lst)}\n")
for percent_noise in percentage:
    no_noise = int(max_points * percent_noise)
    true_data = deepcopy(data)
    mutated_data, unchanged, mutated = perturbedPoints(data, percent_noise, is_random=True)


    for dim in dimension:
        est_dim_lst = []
        for _ in range(no_repeat):
            _, _, est_dim, _ = estimatePersistentHomologyDimension(mutated_data, dim, alpha, max_sampling_size, step_size,
                                                                   no_samples)
        with open("results/PHDimReport/PHDimPointNoise/Sierpinski2D.txt", 'a') as file:
            file.write(f"{max_points}, {percent_noise}, {dim}, {np.mean(est_dim_lst)}, {np.std(est_dim_lst)}\n")
    data = true_data