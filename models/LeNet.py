import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(6, 16, kernel_size= 5),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return x
