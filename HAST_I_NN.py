# HAST I NN. Based on HAST I from https://ieeexplore.ieee.org/document/8171733
from abc import ABC

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f


class HAST_I(nn.Module, ABC):
    def __init__(self, packets):
        super(HAST_I, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 32)  # 128 Channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 32, 4)
        if packets == 192:
            self.fc1 = nn.Linear(97664, 128)
        elif packets == 128:
            self.fc1 = nn.Linear(64896, 128)
        elif packets == 100:
            self.fc1 = nn.Linear(50560, 128)
        elif packets == 95:
            self.fc1 = nn.Linear(48000, 128)
        else:
            raise AttributeError("This amount of packets Haven't been tested")
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)  # Changed to 2 classes

    def forward(self, x, packets):
        reshaped = False
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        if packets == 192:
            x = x.view(-1, 97664)
        elif packets == 128:
            x = x.view(-1, 64896)
        elif packets == 100:
            # Case of size problems due to using the last 100 packets
            if x.shape[3] < 790:
                # print(x.shape)
                miss_feat = 790 - x.shape[3]
                y = x.float()
                z = torch.tensor(np.zeros([1, 32, 2, miss_feat])).float()
                x = torch.cat((y, z), dim=3)
            x = x.view(-1, 50560)
        elif packets == 95:
            x = x.view(-1, 48000)
        else:
            raise AttributeError("This amount of packets Haven't been tested")
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
