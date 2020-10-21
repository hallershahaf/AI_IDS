# First iteration of AI IDS RDP Project
# Written By Idan Tau & Shahaf Haller
# CNN module based on https://ieeexplore.ieee.org/document/8171733
# "Algorithm 1 Spatial Feature Learning" - HAST_I

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as udata
import torch.optim as optim
import numpy as np
import AI_IDS.create_dataset as ds
import os
from AI_IDS.parsing_scripts.random_input import random_input


def num_flat_features(z):
    # All dimensions except the batch dimension
    size = z.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):

    # Actual parameters of the CNN are not mentioned in the article,
    # but since both temp vectors are concatenated the numbers should be 
    # different so both CNN won't be symmetrical
    
    # C{i} - Num of convolution channels
    # S{i} - Size of convolution matrix
    # T{i} - size of maxPooling layer matrix
    # D{i} - size of dense layer
    C = np.array([32,  64])
    S = np.array([[4, 6, 12], [2, 3, 18]])
    T = np.array([3, 3])
    D = np.array([1024, 5])
    
    # These are changed based on how the images are made
    # from packets of data.
    # Current stream size used in the project is 32x48x100.
    # input_channels = 3
    # image_dimension = 32
    mtu = 1514
    cols = 32
    packets = 100
    rows = int(np.ceil(mtu / cols))

    def __init__(self):     
        super(Net, self).__init__()
        
        # Define the layers up to the dense functions:
        # 6 convolutions are used, 2 for each vector
        self.conv_lay1_1 = nn.Conv3d(Net.packets, Net.C[0], Net.S[0], stride=1)
        self.relu_lay1_1 = nn.ReLU()
        self.max_pool2_1 = nn.MaxPool3d(Net.T[0])
        self.conv_lay3_2 = nn.Conv3d(Net.C[0], Net.C[1], Net.S[1], stride=1)
        self.relu_lay3_2 = nn.ReLU()
        self.max_pool4_2 = nn.MaxPool3d(Net.T[1])

        # A convolution matrix scales the image down by (matrix size) - 1;
        conv_reduction = (Net.S - 1)

        # First convolution reduction
        size_1 = Net.rows - conv_reduction[0, 0]
        size_2 = Net.cols - conv_reduction[0, 1]
        size_3 = Net.packets - conv_reduction[0, 2]
        # First Max pooling reduction
        # P.S. make sure the result is even for easier living
        size_1 //= Net.T[0]
        size_2 //= Net.T[0]
        size_3 //= Net.T[0]
        # Second convolution reduction
        size_1 -= conv_reduction[1, 0]
        size_2 -= conv_reduction[1, 1]
        size_3 -= conv_reduction[1, 2]
        # Second Max pooling reduction
        # P.S. make sure the result is even for easier living        
        size_1 //= Net.T[1]
        size_2 //= Net.T[1]
        size_3 //= Net.T[1]

        # Final size for the linear vector reduction process
        # The size needs to be x * y * final channels
        size_1 = (size_1 ** 2) * Net.C[1]
        size_2 = (size_2 ** 2) * Net.C[1]
        size_3 = (size_3 ** 2) * Net.C[1]
        # First linear reduction
        self.fc1_1 = nn.Linear(size_1 // 4, size_1 // 16)
        self.fc2_1 = nn.Linear(size_2 // 4, size_2 // 16)
        self.fc3_1 = nn.Linear(size_3 // 4, size_3 // 16)
        # Second linear reduction
        self.fc1_2 = nn.Linear(size_1 // 16, size_1 // 32)
        self.fc2_2 = nn.Linear(size_2 // 16, size_2 // 32)
        self.fc3_2 = nn.Linear(size_3 // 16, size_3 // 32)

        # Final linear reduction reduces us to a [false, positive] vector
        self.fc1_3 = nn.Linear(size_1 // 32, 2)
        self.fc2_3 = nn.Linear(size_2 // 32, 2)
        self.fc3_3 = nn.Linear(size_3 // 32, 2)

    def forward(self, z):
        z = torch.tensor(z)
        # relu(x) -> max{x,0}
        print("tensor")
        # Conv -> relu -> pool -> conv -> relu -> pool
        X = self.max_pool2_1(self.relu_lay1_1(self.conv_lay1_1(z)), Net.T[0])
        X = self.max_pool4_2(self.relu_lay3_2(self.conv_lay3_2(X)), Net.T[1])

        Y = self.max_pool2_1(self.relu_lay1_1(self.conv_lay1_1(z)), Net.T[0])
        Y = self.max_pool4_2(self.relu_lay3_2(self.conv_lay3_2(Y)), Net.T[1])

        Z = self.max_pool2_1(self.relu_lay3_2(self.conv_lay3_2(z)), Net.T[1])
        Z = self.max_pool4_2(self.relu_lay1_1(self.conv_lay1_1(Z)), Net.T[0])

        # Turn the matrices into long vectors
        X = X.view(1, num_flat_features(X))
        Y = Y.view(1, num_flat_features(Y))
        Z = Z.view(1, num_flat_features(Z))
        # Reduce the vectors
        X = func.relu(self.fc1_1(X))
        X = func.relu(self.fc2_1(X))
        X = self.fc3_1(X)
        
        Y = func.relu(self.fc1_2(Y))
        Y = func.relu(self.fc2_2(Y))
        Y = self.fc3_2(Y)

        Z = func.relu(self.fc1_3(Z))
        Z = func.relu(self.fc2_3(Z))
        Z = self.fc3_3(Z)

        return X + Y + Z


net = Net()
print(net)

epoches = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


train_dir = os.path.join(os.getcwd(), "Dataset")
train_set = ds.create_dataset(train_dir, "eOs.npy")
train_loader = udata.DataLoader(train_set)

for epoch in range(epoches):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data[inputs])
        loss = criterion(outputs, data[labels])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0

print('Finished Training')
