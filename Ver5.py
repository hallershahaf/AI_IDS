# Fourth version - Random dataset has same struct as ours
import os
import AI_IDS.create_dataset_v2 as dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# pytorch_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
# pytorch_trainloader = torch.utils.data.DataLoader(pytorch_trainset, batch_size=4,
#                                           shuffle=True, num_workers=0)

train_dir = os.path.join(os.getcwd()[:-6], "Dataset")
train_set = dataset.create_dataset(train_dir, "EoS.npy")
trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
test_dir = os.path.join(os.getcwd()[:-6], "Datatest")
test_set = dataset.create_dataset(test_dir, "EoS.npy")
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


classes = ('safe', 'exploit')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(100, 100, 8)     # 100 Channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 16, 8)
        self.fc1 = nn.Linear(16 * 3 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)     # Changed to 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # if not((predicted == labels)) and labels[0] == 0:
        #     print(predicted, " -> ", labels)
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            optimizer.step()
            accuracy = 100.0 * correct / total
            print('accuracy = %.2f %%' % accuracy)
            running_loss = 0.0

print('Finished Training')
print('Starting Test')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) -1)  ,'test images: %d %%' % (
    100 * correct / total))