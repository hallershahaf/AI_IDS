# Run HAST I from an outside class
import os
# self made functions and classes
import AI_IDS.create_dataset_v2 as dataset
from AI_IDS.HAST_I_NN import HAST_I

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# import torchvision


# Testing the network on CIFAR10
# pytorch_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
# pytorch_trainloader = torch.utils.data.DataLoader(pytorch_trainset, batch_size=4,
#                                           shuffle=True, num_workers=0)

# Defining the data for the NN
train_dir = os.path.join(os.getcwd()[:-6], "Dataset_HAST_I_0_big")
train_set = dataset.create_dataset(train_dir, "EoS.npy")
trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0")
test_set = dataset.create_dataset(test_dir, "EoS.npy")
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

classes = ('safe', 'exploit')

# Defining the NN
net = HAST_I()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Training the NN
epochs = 5
packets = 128
mtu = 1514
cols = 32
rows = 48
safe_sign = 3

for epoch in range(epochs):  # loop over the dataset multiple times
    print(epoch)
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

# Saving state
print('Saving state')
torch.save({
            'epoch': epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss': loss
            }, os.path.join(os.getcwd(), "metasploit_post-training_NN", "NN_post_training_HAST_I_5_e"))


# Testing the NN
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

print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
        100 * correct / total))
