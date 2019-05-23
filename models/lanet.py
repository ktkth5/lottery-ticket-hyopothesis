import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_module import PruningModule, MaskedLinear, MaskedConv2d
import torch.nn.init as init

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(28*28, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, MaskedLinear)):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = conv(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = conv(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = linear(64*16*16, 256)
        self.fc2 = linear(256,256)
        self.fc3 = linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, MaskedConv2d):
                # nn.init.kaiming_normal_(m.weight_prune, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight_prune)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, MaskedLinear)):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Conv1
        x = F.relu(self.conv1(x)) # ; print("x", x.shape)

        # Conv2
        x = F.relu(self.conv2(x)) # ; print("x", x.shape)
        x = self.pool(x) # ; print("x", x.shape)

        # Fully-connected
        x = x.view(-1, 64*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)

        return x

    def reinit(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                # nn.init.kaiming_normal_(m.weight_prune, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight_prune)
                m.weight_prune.data = m.weight_prune.data * m.mask.data

            elif isinstance(m, MaskedLinear):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_uniform_(m.weight)
                m.weight.data = m.weight.data * m.mask.data


if __name__=="__main__":
    x = torch.randn(2,3,32,32)
    model = LeNet_5(mask=True)
    y = model(x)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

