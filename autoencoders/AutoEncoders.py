from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

class VAE1(nn.Module):
    def __init__(self, imageChannels=1, featureDim=32 * 20 * 20, zdim=2):
        super(VAE1, self).__init__()

        self.enConv1 = nn.Conv2d(imageChannels, 16, 5)
        self.enConv2 = nn.Conv2d(16, 32, 5)
        self.enFC1 = nn.Linear(featureDim, zdim)
        self.enFC2 = nn.Linear(featureDim, zdim)

        self.deFC1 = nn.Linear(zdim, featureDim)
        self.deConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.deConv2 = nn.ConvTranspose2d(16, imageChannels, 5)

    def encoder(self, x):
        x = F.relu(self.enConv1(x))
        x = F.relu(self.enConv2(x))
        x = x.view(-1, 32 * 20 * 20)
        mu = self.enFC1(x)
        logVar = self.enFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar / 2)
        eps = torch.rand_like(std)

        return mu + eps * std

    def decoder(self, z):
        x = F.relu(self.deFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.deConv1(x))
        x = torch.sigmoid(self.deConv2(x))
        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

class VAE2(nn.Module):
    def __init__(self, image_channels=1, latent_dim=10):
        super(VAE2, self).__init__()

        self.en_conv1 = nn.Conv2d(image_channels, 16, 3, padding="same")
        self.en_conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.en_conv3 = nn.Conv2d(32, 32, 3, padding="same")

        self.en_fc1 = nn.Linear(28 * 28 * 32, 32)

        self.mu = nn.Linear(32, latent_dim)
        self.log_var = nn.Linear(32, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 32)
        self.de_fc2 = nn.Linear(32, 28 * 28 * 32)

        self.de_conv1 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.de_conv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(16, image_channels, 3, padding=1)

    def encoder(self, x):
        x = F.relu(self.en_conv1(x))
        x = F.relu(self.en_conv2(x))
        x = F.relu(self.en_conv3(x))
        x = nn.Flatten(start_dim=1)(x)
        x = F.relu(self.en_fc1(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.rand_like(std)

        return mu + eps * std

    def decoder(self, z):
        z = F.relu(self.de_fc1(z))
        z = F.relu(self.de_fc2(z))
        z = z.view(-1, 32, 28, 28)
        z = F.relu(self.de_conv1(z))
        z = F.relu(self.de_conv2(z))
        z = torch.sigmoid(self.de_conv3(z))
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var


if __name__ == "__main__":
    batch_size = 100
    learning_rate = 1e-3
    num_epochs = 50
    device = "cpu"
    image_channels = 1
    latent_dim = 12
    type_model = 2

    epoch_str_lst = []
    model_save_path = "./vae/VAE"+str(type_model)+"/vae_hidden"+str(latent_dim)+".pt"
    info_save_path = "./vae/VAE"+str(type_model)+"/vae_hidden"+str(latent_dim)+".txt"
    print(model_save_path)
    print(info_save_path)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False)
    if type_model == 1:
        vae = VAE1(1, 32 * 20 * 20, latent_dim).to(device)
    elif type_model == 2:
        vae = VAE2(image_channels=image_channels, latent_dim=latent_dim).to(device)
    else:
        raise ValueError("Model not supported")
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        start_time = time.time()
        for idx, data in enumerate(test_loader, 0):
            imgs, _ = data
            imgs = imgs.view(-1, 1, 28, 28)
            imgs = imgs.float()
            imgs = imgs.to(device)
            out, mu, logVar = vae(imgs)

            # loss function
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            recon_loss = F.binary_cross_entropy(out, imgs, reduction="sum")
            loss = recon_loss + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        exec_time = time.time() - start_time
        epoch_str = f"Epoch {epoch}|Total loss: {loss}|Reconstruction loss: {recon_loss}|KL divergence: {kl_divergence}|Epoch time: {exec_time}"
        epoch_str_lst.append(epoch_str)
        print(epoch_str)

    torch.save(vae.state_dict(), model_save_path)
    with open(info_save_path, 'a') as file:
        for info in epoch_str_lst:
            file.write(info+"\n")
