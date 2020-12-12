# HAST I NN. Based on HAST I from https://ieeexplore.ieee.org/document/8171733
from abc import ABC

import torch.nn as nn
import torch.nn.functional as f


class HAST_I(nn.Module, ABC):
    def __init__(self):
        super(HAST_I, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 32)  # 128 Channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 32, 4)
        self.fc1 = nn.Linear(50560, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)  # Changed to 2 classes

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 50560)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
