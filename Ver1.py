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
import create_dataset as ds
import os


def num_flat_features(z):
    # All dimensions except the batch dimension
    size = z.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# noinspection PyAbstractClass
class Net(nn.Module):

    # Actual parameters of the CNN are not mentioned in the article,
    # but since both temp vectors are concatenated the numbers should be 
    # different so both CNN won't be symmetrical
    
    # C{i} - Num of convolution channels
    # S{i} - Size of convolution matrix
    # T{i} - size of maxPooling layer matrix
    C = np.array([6,  16,  6,  16])
    S = np.array([3,   4,  5,   3])
    T = np.array([2,   2,  2,   2])
    
    # These are changed based on how the images are made
    # from packets of data.
    # Current image size used in the project is 32x48.
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu/cols))   # 48
    input_channels = 100    # Number of packets per stream

    def __init__(self):     
        super(Net, self).__init__()
        
        # Define the convolution functions:
        # 4 convolutions are used, 2 for each vector.
        self.conv1_1 = nn.Conv2d(Net.input_channels, Net.C[0], Net.S[0])
        self.conv1_2 = nn.Conv2d(Net.C[0], Net.C[1], Net.S[1])
        self.conv2_1 = nn.Conv2d(Net.input_channels, Net.C[2], Net.S[2])
        self.conv2_2 = nn.Conv2d(Net.C[2], Net.C[3], Net.S[3])

        # The size of the images (not including depth/channels)
        size_1 = [Net.cols, Net.rows]
        size_2 = [Net.cols, Net.rows]

        # A convolution matrix scales the image down by (matrix size) - 1;
        conv_reduction = (Net.S - 1)
        
        # First convolution reduction
        size_1 -= conv_reduction[0]
        size_2 -= conv_reduction[2]
        
        # First Max pooling reduction
        # P.S. make sure the result is even for easier living
        size_1 //= Net.T[0]
        size_2 //= Net.T[2]
        
        # Second convolution reduction
        size_1 -= conv_reduction[1]
        size_2 -= conv_reduction[3]
        
        # Second Max pooling reduction
        # P.S. make sure the result is even for easier living        
        size_1 //= Net.T[1]
        size_2 //= Net.T[3]
        
        # Final size for the linear vector reduction process
        # The size needs to be height * width * final channels
        size_1 = size_1[0] * size_1[1] * Net.C[1]
        size_2 = size_2[0] * size_2[1] * Net.C[3]

        # First linear reduction
        self.fc1_1 = nn.Linear(size_1, size_1 // 2)
        self.fc2_1 = nn.Linear(size_2, size_2 // 2)
        
        # Second linear reduction
        self.fc1_2 = nn.Linear(size_1 // 2, size_1 // 4)
        self.fc2_2 = nn.Linear(size_2 // 2, size_2 // 4)

        # Final linear reduction reduces us to a [false, positive] vector
        self.fc1_3 = nn.Linear(size_1 // 4, 2)
        self.fc2_3 = nn.Linear(size_2 // 4, 2)

    def forward(self, z):

        # x/y -> First/second path
        # relu(x) -> max{x,0}
        
        # Conv -> relu -> pool -> conv -> relu -> pool
        x = func.max_pool2d(func.relu(self.conv1_1(z)), Net.T[0])
        x = func.max_pool2d(func.relu(self.conv1_2(x)), Net.T[1])
    
        y = func.max_pool2d(func.relu(self.conv2_1(z)), Net.T[2])
        y = func.max_pool2d(func.relu(self.conv2_2(y)), Net.T[3])

        # Turn the matrices into long vectors
        x = x.view(1, num_flat_features(x))
        y = y.view(1, num_flat_features(y))
        
        # Reduce the vectors
        x = func.relu(self.fc1_1(x))
        x = func.relu(self.fc1_2(x))
        x = self.fc1_3(x)
        
        y = func.relu(self.fc2_1(y))
        y = func.relu(self.fc2_2(y))
        y = self.fc2_3(y)

        # Divide by 2 to normalize
        return (x + y) / 2


verbose = False
verbose_step = 25
net = Net()
print(net)

# The system needs to run on a single type or bad things happen
net = net.float()

epochs = 10  # TODO change epochs back to big number
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
softmax = nn.Softmax(dim=1)

train_dir = os.path.join(os.getcwd(), "Dataset")
train_set = ds.create_dataset(train_dir, "EoS.npy")
train_loader = udata.DataLoader(train_set)


for epoch in range(epochs):  # loop over the dataset multiple times

    correct = 0
    total = 0

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        # The system needs to run on a single type or bad things happen
        output = softmax(net(data[inputs].float()))
        # Verbose 1 - results vs expected
        if verbose:
            print(output[0].detach(), " -> ", int(data[labels][0]))
        # Verbose 2 - percentage
        total += 1
        if torch.max(output, 1)[1] == data[labels].long():
            correct += 1
        # Labels are of type Long, not float
        loss = criterion(output, data[labels].long())
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % verbose_step == 0 and verbose:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / verbose_step))
            running_loss = 0.0

    print("Percentage ->", (correct / total) * 100, "%")
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.3

print('Finished Training')
