from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F



class FC(nn.Module):

    # 28 x 28 for MNIST dataset
    def __init__(self, input_dim = 28 * 28, input_width = 50, depth = 3, no_class = 10, active_bias = True):
        super(FC, self).__init__()

        self.input_dim = input_dim
        self.input_width = input_width
        self.depth = depth
        self.no_class = no_class
        self.active_bias = active_bias
        self.selected_output = OrderedDict()

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, self.input_width, bias=self.active_bias),
            nn.ReLU(inplace=True),
            *self.get_layers(),
            nn.Linear(self.input_width, self.no_class, bias=self.active_bias),
        )

    def get_layers(self):
        layers = []

        for _ in range(self.depth - 2):
            # -2 for input and output layers
            layers.append(nn.Linear(self.input_width, self.input_width, bias=self.active_bias))
            layers.append(nn.ReLU())

        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.features(x)
        return x

    def get_information_str(self):
        return f"FC(Width:{self.input_width},Depth:{self.depth})"

    def get_features(self, layer_name):
        def hook(module, input, output):
            self.selected_output[layer_name] = output.detach()
        return hook


# def fc(**kwargs):
#     return FC(**kwargs)
