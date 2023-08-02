import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as transforms
from Interpolate import generateInterpolate
from models.AlexNet import AlexNet
from torch import nn
import matplotlib.pyplot as plt
import random


class InterpolateDataset(Dataset):
    def __init__(self, images, labels, img_height=28, img_width=28, img_channel=1):
        self.data = torch.from_numpy(images).view(-1, img_channel, img_width, img_height)
        self.targets = torch.from_numpy(labels)

    def __getitem__(self, idex):
        img = self.data[idex]
        labels = self.targets[idex]
        return img, labels

    def __len__(self):
        return len(self.targets)

def evalModel(model, test_loader):
    model.eval()

    # Softmax for probability-liked
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        total_point = 0
        correct = 0
        true_label_lst = []
        wrong_pred_label_lst = []
        data_lst = []
        confidence_lst = []
        for input_batch, label_batch in test_loader:
            input_batch, label_batch = input_batch.to(device), label_batch.to(device)
            output = model(input_batch)
            output = softmax(output)
            pred = torch.max(output.data, 1)[1]
            correct += (pred == label_batch).sum().item()
            total_point += label_batch.size(0)

            output = output.cpu().numpy()
            pred = pred.cpu().numpy()
            target = label_batch.cpu().numpy()
            pred = np.reshape(pred, (len(pred), 1))
            target = np.reshape(target, (len(pred), 1))
            dat = input_batch.cpu().numpy()

            for i in range(len(pred)):
                # Wrong prediction
                if (pred[i] != target[i]):
                    wrong_pred_label_lst.append(pred[i])
                    true_label_lst.append(target[i])
                    data_lst.append(dat[i])
                    confidence_lst.append(output[i])
    acc = correct / total_point
    return acc, wrong_pred_label_lst, true_label_lst, data_lst, confidence_lst

def plotWrongImage(wrong_label_list, true_label_list, images_list, confidence_list, no_plot, title_str, save_path):
    if no_plot > 50:
        no_plot = 50
    no_col = 5
    no_row = int(no_plot / no_col)
    fig, axes = plt.subplots(figsize=(15, 20), nrows=no_row, ncols=no_col)
    # fig.tight_layout()
    if no_plot > len(wrong_label_list): no_plot = len(wrong_label_list)
    chosen_idx = random.sample(range(len(wrong_label_list)), no_plot)
    for idx, ax in enumerate(axes.flatten()):
        if idx > no_plot-1:
            break
        pred_label, true_label, imag, confidence = wrong_label_list[chosen_idx[idx]], true_label_list[chosen_idx[idx]], images_list[chosen_idx[idx]], confidence_list[chosen_idx[idx]]
        top_3_idx = np.argsort(confidence)[::-1][:3]
        top_3_val = confidence[top_3_idx]
        img_title = "Pred:" + str(pred_label[0]) + " True:"+str(true_label[0])
        for i in range(3):
            img_title += "\nClass " + str(top_3_idx[i]) + ": "+str(top_3_val[i])
        im = ax.imshow(imag.transpose(1,2,0), cmap="gray")
        ax.set_title(img_title)
        ax.axis('off')
    fig.suptitle(title_str)
    plt.savefig(save_path)
    plt.close(fig)
    # plt.show()


dataset_name = "MNIST"
batch_size = 1000
device = "cpu"
if dataset_name == "MNIST":
    img_width = 28
    img_height = 28
    img_channels = 1
    no_class = 10

    vae_save_path = "./vae/VAE2/vae_hidden12.pt"
    model_save_path = "./results/TrainedModels/AlexNet_MNIST/AlexNetRobust1.pth"

    # synthesis_img, target = generateInterpolate(img_channels, vae_save_path, batch_size, no_class)

    synthesis_img = np.load("tmp/syn_img.npy")
    target = np.load("tmp/target.npy")
    # Load data
    original_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    gen_data = InterpolateDataset(synthesis_img, target, img_height, img_width, img_channels)

    # Load model
    model = AlexNet(input_height=img_height, input_width=img_width, input_channels=img_channels, no_class=no_class)
    model.load_state_dict(torch.load(model_save_path))

for idx in range(no_class):
    test_idx = np.where((np.array(original_data.targets) == idx))[0]
    test_data_1_class = Subset(original_data, test_idx)
    original_test_loader = DataLoader(dataset=test_data_1_class, batch_size=batch_size, shuffle=False)

    test_idx = np.where((np.array(gen_data.targets) == idx))[0]
    test_data_1_class = Subset(gen_data, test_idx)
    gen_data_test_loader = DataLoader(dataset=test_data_1_class, batch_size=batch_size, shuffle=False)

    acc, wrong_pred_label_lst, true_label_lst, data_lst, confidence_lst = evalModel(model, original_test_loader)
    acc_synthesis, wrong_pred_label_lst_sythesis, true_label_lst_synthesis, data_lst_sythesis, synthesis_confidence_lst = evalModel(
        model, gen_data_test_loader)

    save_path_1 = "results/AccSynthesisImgs/MNIST/AlexNet/Robust/originalClass"+str(idx)+".png"
    save_path_2 = "results/AccSynthesisImgs/MNIST/AlexNet/Robust/synthesisClass"+str(idx)+".png"
    plotWrongImage(wrong_pred_label_lst, true_label_lst, data_lst, confidence_lst, 20,
                   "AlexNet Robust on MNIST\nAccuracy:" + str(acc), save_path_1)
    plotWrongImage(wrong_pred_label_lst_sythesis, true_label_lst_synthesis, data_lst_sythesis, synthesis_confidence_lst,
                   20, "AlexNet Robust on MNIST Interpolate\nAccuracy:" + str(acc_synthesis), save_path_2)

# original_test_loader = DataLoader(original_data, batch_size=batch_size, shuffle=False)
# gen_data_test_loader = DataLoader(gen_data, batch_size=batch_size, shuffle=False)


# acc, wrong_pred_label_lst, true_label_lst, data_lst, confidence_lst = evalModel(model, original_test_loader)
# acc_synthesis, wrong_pred_label_lst_sythesis, true_label_lst_synthesis, data_lst_sythesis, synthesis_confidence_lst = evalModel(model, gen_data_test_loader)
#
# save_path_1 = "results/AccSynthesisImgs/MNIST/AlexNet/Normal/original.png"
# save_path_2 = "results/AccSynthesisImgs/MNIST/AlexNet/Normal/synthesis.png"
# plotWrongImage(wrong_pred_label_lst, true_label_lst, data_lst, confidence_lst, 20, "AlexNet Robust on MNIST\nAccuracy:"+str(acc), save_path_1)
# plotWrongImage(wrong_pred_label_lst_sythesis, true_label_lst_synthesis, data_lst_sythesis, synthesis_confidence_lst, 20, "AlexNet Robust on MNIST Interpolate\nAccuracy:"+str(acc_synthesis), save_path_2)
# print(acc)
# print(acc_synthesis)
# np.save("tmp/syn_img", synthesis_img)
# np.save("tmp/target", target)

