from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from scipy.spatial import distance, cKDTree
from autoencoders.AutoEncoders import VAE1, VAE2
from ComputeTopoDescriptors import *
np.random.seed(42)

def generateImage(model, save_path, latent_dim=12):
    z = torch.randn(50, latent_dim)
    synth_data = model.decoder(z).cpu().detach().numpy().reshape(-1, 28, 28, 1)

    plt.figure(figsize=(8, 7))
    for i in range(50):
        # 5 rows and 10 columns-
        plt.subplot(5, 10, i + 1)
        plt.imshow(synth_data[i], cmap='gray')

        # get current axes-
        ax = plt.gca()

        # hide x-axis-
        ax.get_xaxis().set_visible(False)

        # hide y-axis-
        ax.get_yaxis().set_visible(False)

    plt.suptitle("VAE Synthesized images for MNIST")
    plt.savefig(save_path)
    plt.show()
    plt.clf()

# Build KD tree to find k-nearest neighbors of a point
def nearestNeighbors(arr, k):
    neighbors_idx_lst = []
    neighbors_dist_lst = []
    k_list = range(2, k+2)
    kd_tree = cKDTree(arr)
    for i, ele in enumerate(arr):
        dist, idx = kd_tree.query(ele, k=k_list, workers=2)
        neighbors_idx_lst.append(list(idx))
        neighbors_dist_lst.append(list(dist))
    return neighbors_idx_lst, neighbors_dist_lst

def generateInterpolate(image_channels, model_save_path, batch_size, no_class):
    # image_channels = 1
    latent_dim = 12
    type_model = 2
    # path_save_imgs = f"./plots/Conv_VAE/VAE2/synthesis_hidden{latent_dim}.png"
    # dataset_name = "MNIST_test"
    # path_save_data_decoder = f"./info/{dataset_name}/decoder.txt"
    # path_save_data_latent = f"./info/{dataset_name}/latent.txt"
    # no_class = 10

    # model_save_path = "./vae/VAE" + str(type_model) + "/vae_hidden" + str(latent_dim) + ".pt"

    vae = VAE2(image_channels=image_channels, latent_dim=latent_dim).to("cpu")
    vae.load_state_dict(torch.load(model_save_path))
    # vae.double()
    # vae = VAE1().to("cpu")

    vae.eval()

    # batch_size = 1000
    device = "cpu"

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False)

    mu_list = []
    logvar_list = []
    target_labels = []
    recon_img_lst = []

    for input_batch, target_batch in test_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        recon_img, mu, log_var = vae(input_batch)
        mu = mu.cpu().detach().numpy()
        log_var = log_var.cpu().detach().numpy()
        recon_img = recon_img.view(batch_size, -1).cpu().detach().numpy()

        recon_img_lst.append(recon_img)
        mu_list.append(mu)
        logvar_list.append(log_var)
        target_labels.append(target_batch.numpy())

    mu_list = np.vstack(mu_list)
    logvar_list = np.vstack(logvar_list)
    target_labels = np.vstack(target_labels).flatten()


    latent_vec = mu_list + np.exp(0.5 * logvar_list) * np.random.rand(*mu_list.shape)
    # Variables for expanding
    total_latent_vec_lst = np.asarray(latent_vec)
    recon_img_lst = np.vstack(recon_img_lst)
    # print(total_latent_vec_lst.shape)
    # print(recon_img_lst.shape)
    #
    # breakpoint()
    # evalDataBatch(dataset_name + f"|VAE_latent{latent_dim}", recon_img_lst, path_save_data_decoder, batch_size=1000, no_neighbors=100,
    #               metric="euclidean")
    #
    # evalDataBatch(dataset_name + f"|VAE_latent{latent_dim}", latent_vec, path_save_data_latent, batch_size=1000, no_neighbors=100,
    #               metric="euclidean")

    latent_idx_lst = []
    latent_vec_lst = []
    mean_latent_vec_lst = []
    dist_lst = []
    quantile_dist_lst = []
    quantile_idx_lst = []
    quantile_vec_lst = []
    exclude_latent_idx_lst = []
    exclude_latent_vec_lst = []
    quantile = 0.7

    for i in range(no_class):
        latent_idx = np.where(target_labels == i)[0]
        latent_vec_class = latent_vec[latent_idx, :]
        centroid_class = np.mean(latent_vec_class, axis=0)

        # distance to the centroid of each class
        dist_centroid_class = np.linalg.norm(centroid_class - latent_vec_class, axis=1)

        # Find the quantile value of the distances
        quantile_dist_class = np.quantile(np.sort(dist_centroid_class), quantile)
        # Find all idx within the quantile
        quantile_idx_class = np.where(dist_centroid_class <= quantile_dist_class)[0]

        # Find all latent vectors from the indices
        quantile_vec_class = latent_vec_class[quantile_idx_class]
        # Find the rest of unchosen index
        exclude_latent_idx_class = np.setdiff1d(range(len(latent_vec_class)), quantile_idx_class)
        # Find the excluded latent vectors
        exclude_latent_vec_class = latent_vec_class[exclude_latent_idx_class]

        latent_idx_lst.append(latent_idx)
        latent_vec_lst.append(latent_vec_class)
        mean_latent_vec_lst.append(centroid_class)
        quantile_idx_lst.append(quantile_idx_class)
        quantile_dist_lst.append(quantile_dist_class)
        quantile_vec_lst.append(quantile_vec_class)
        exclude_latent_idx_lst.append(exclude_latent_idx_class)
        exclude_latent_vec_lst.append(exclude_latent_vec_class)

    # # Each rows contains indices of k-nearest neighbors to the point whose index is the current row
    # no_neighbors = 9
    # nearest_idx_lst, nearest_dist_lst = nearest_neighbors(mean_latent_vec_lst, no_neighbors)

    # Choose randomly some points
    eps1 = 0.05
    eps2 = 0.05
    percentage_sampling = 0.3
    # Interpolate type: 0: using mean, 1: using 1 point inside, 2: using point outside
    interpolate_type = 0
    # Interpolate
    interpolate_ratio_lst = [0.1, 0.2, 0.3]

    even_spread_lst = [(1 - latent_idx_lst[i].shape[0]/len(test_loader.dataset)) for i in range(no_class)]
    list_class_gen_more = np.where(np.array(even_spread_lst) > eps1)[0]
    for curr_class in list_class_gen_more:
        centroid_point = len(quantile_idx_lst[curr_class])/len(latent_idx_lst[curr_class])
        if centroid_point < eps2:
            continue

        no_sampling = int(percentage_sampling * len(exclude_latent_idx_lst[curr_class]))

        chosen_exclude_vec_idx = np.random.choice(exclude_latent_idx_lst[curr_class], no_sampling)
        chosen_exclude_vec = latent_vec_lst[curr_class][chosen_exclude_vec_idx]

        # Find the closest class that the point are outside of the
        # Find distance from this point to all centroids
        dist_to_centroid = [np.linalg.norm(chosen_vec - mean_latent_vec_lst, axis=1) for chosen_vec in chosen_exclude_vec]
        # Delete the current class as it we want to create ambiguities from this class
        dist_to_centroid = [np.delete(dist, curr_class) for dist in dist_to_centroid]
        # Find the difference between the point and the radius of each class

        diff_radii = np.vstack(dist_to_centroid) - np.delete(np.vstack(quantile_dist_lst), curr_class)
        # Get the smallest value's index
        class_interpolate_idx = np.argmin(diff_radii, axis=1)
        # Increase 1 for all value greater or equal to deleted_class
        class_interpolate_idx[class_interpolate_idx >= curr_class] += 1
        print(f"Class {curr_class}")
        # print(len(class_interpolate_idx))
        for i, class_idx in enumerate(class_interpolate_idx):
            for ratio in interpolate_ratio_lst:
                # new_latent = chosen_0 + dir_01 * ratio
                if interpolate_type == 0:
                    # Using mean latent vector of class
                    new_latent = (1-ratio) * chosen_exclude_vec[i,:] + ratio * mean_latent_vec_lst[class_idx]
                elif interpolate_type == 1:
                    # Using one randomly chosen vector inside quantile
                    interpolate_chosen_idx = np.random.choice(range(quantile_vec_lst[class_idx].shape[0]), 1)
                    interpolate_chosen_vec = quantile_vec_lst[class_idx][interpolate_chosen_idx]
                    new_latent = (1-ratio) * chosen_exclude_vec[i,:] + ratio * interpolate_chosen_vec
                elif interpolate_type == 2:
                    interpolate_chosen_idx = np.random.choice(range(exclude_latent_vec_lst[class_idx].shape[0]), 1)
                    interpolate_chosen_vec = exclude_latent_vec_lst[class_idx][interpolate_chosen_idx]
                    new_latent = (1 - ratio) * chosen_exclude_vec[i, :] + ratio * interpolate_chosen_vec
                else:
                    raise ValueError("Type interpolation not supported")
                # new_latent.astype(np.float32)
                new_latent = torch.from_numpy(new_latent).double()
                new_latent = new_latent.float()
                new_latent_lst_form = new_latent.cpu().detach().numpy()
                total_latent_vec_lst = np.vstack([total_latent_vec_lst, new_latent_lst_form])

                # Decode the new latent
                new_img = vae.decoder(new_latent).cpu().detach().numpy().reshape(-1, 28, 28, 1)
                new_img_lst_form = new_img.reshape(1, -1)
                # print(new_img_lst_form)
                recon_img_lst = np.vstack([recon_img_lst, new_img_lst_form])

                # add new class labels
                target_labels = np.append(target_labels, curr_class)
                # print(target_labels.shape)
                #
                # plt.imshow(new_img[0], cmap='gray')
                #
                # # get current axes-
                # ax = plt.gca()
                #
                # # hide x-axis-
                # ax.get_xaxis().set_visible(False)
                #
                # # hide y-axis-
                # ax.get_yaxis().set_visible(False)
                # class_name = f"{curr_class}" + str() if ratio <= 0.5 else f"{class_idx}"
                # plt.title(class_name+" ratio: "+str(ratio))
                # plt.show()

    # dataset_name = "MNIST"
    # path_save_data_decoder_gen = f"./info/{dataset_name}_gen/decoder.txt"
    # path_save_data_latent_gen = f"./info/{dataset_name}_gen/latent.txt"
    #
    # evalDataBatch(dataset_name + f"|VAE_latent{latent_dim}", recon_img_lst, path_save_data_decoder_gen, batch_size=1000, no_neighbors=100,
    #               metric="euclidean")
    #
    # evalDataBatch(dataset_name + f"|VAE_latent{latent_dim}", total_latent_vec_lst, path_save_data_latent_gen, batch_size=1000, no_neighbors=100,
    #               metric="euclidean")
    return recon_img_lst, target_labels


