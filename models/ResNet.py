from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               stride=stride, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channel)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        if self.down_sample:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, no_block_lst, no_class, is_grayscale):
        self.in_channel = 64
        if is_grayscale:
            in_channel = 1
        else:
            in_channel = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_block(block, 64, no_block_lst[0])
        self.block2 = self._make_block(block, 128, no_block_lst[1], stride=2)
        self.block3 = self._make_block(block, 256, no_block_lst[2], stride=2)
        self.block4 = self._make_block(block, 512, no_block_lst[3], stride=2)
        self.avg_pooling = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=no_class)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Initialize conv weight
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



    def _make_block(self, block, out_channel, no_block, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=out_channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, down_sample))
        self.in_channel = out_channel * block.expansion
        for i in range(1, no_block):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18(no_class, is_gray_scale):
    model = ResNet(BasicBlock, [2, 2, 2, 2], no_class, is_gray_scale)
    return model

def resnet34(no_class, is_gray_scale):
    model = ResNet(BasicBlock, [3, 6, 4, 3], no_class, is_gray_scale)
    return model