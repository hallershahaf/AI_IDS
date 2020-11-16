# Run HAST I from an outside class
import os
# self made functions and classes
import AI_IDS.create_dataset_v2 as dataset
from AI_IDS.HAST_I_NN import HAST_I
from AI_IDS.test_NN_wo_load import test_wo_load as twol

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
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

# test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0")
# test_set = dataset.create_dataset(test_dir, "EoS.npy")
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

classes = ('safe', 'exploit')

# Defining the NN
net = HAST_I()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Training the NN
epochs = 10
packets = 128
mtu = 1514
cols = 32
rows = 48
safe_sign = 3

Datatest_100 = []
Datatest_75 = []
Datatest_50 = []
Datatest_25 = []

for epoch in range(epochs):  # loop over the dataset multiple times
    print("epoch = ", str(epoch))
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # Add noise + move the images
        if epoch == 0 :
            cropped_inputs = inputs[:, :, :, 0:4096]
        else:
            cropped_inputs = inputs[:, :, :, (cols * ((epoch % 3) + 1)):((packets * cols) + (cols * ((epoch % 3) + 1)))]

        # print(cropped_inputs.size())

        # forward + backward + optimize
        outputs = net(cropped_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            optimizer.step()
            accuracy_train = 100.0 * correct_train / total_train
            print('accuracy = %.2f %%' % accuracy_train)
            running_loss = 0.0

    # Test each epoch
    net.eval()
    with torch.no_grad():
        Datatest_100.append(twol("Datatest_HAST_I_0", net.state_dict()))
        Datatest_75.append(twol("Datatest_HAST_I_diff_safe&safe", net.state_dict()))
        Datatest_50.append(twol("Datatest_HAST_I_0xeb&safe", net.state_dict()))
        Datatest_25.append(twol("Datatest_HAST_I_0xeb&safe&diff_safe", net.state_dict()))
    net.train()

print('Finished Training')

# # Saving state
# print('Saving state')
# torch.save({
#             'epoch': epochs,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict':optimizer.state_dict(),
#             'loss': loss
#             }, os.path.join(os.getcwd(), "metasploit_post-training_NN", "NN_post_training_HAST_I_6_e_V2"))


# Printing the Statistics
print("Printings statistics")
plt.plot(np.arange(1, len(Datatest_100) + 1), Datatest_100, label="100% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_75) + 1), Datatest_75, label="75% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_50) + 1), Datatest_50, label="50% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_25) + 1), Datatest_25, label="25% similar", marker='o')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(np.arange(1, epochs + 1, 1))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Vs. Epochs')
plt.legend()
plt.show()



# # Testing the NN
# test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0")
# test_set = dataset.create_dataset(test_dir, "EoS.npy")
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

# print('Starting Test', test_dir)
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
#         100 * correct / total))
#
# test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_diff_safe&safe")
# test_set = dataset.create_dataset(test_dir, "EoS.npy")
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
#
# # Testing the NN
# print('Starting Test', test_dir)
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
#         100 * correct / total))
#
#
# test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0xeb&safe")
# test_set = dataset.create_dataset(test_dir, "EoS.npy")
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
#
# # Testing the NN
# print('Starting Test', test_dir)
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
#         100 * correct / total))
#
# test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0xeb&safe&diff_safe")
# test_set = dataset.create_dataset(test_dir, "EoS.npy")
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
#
# # Testing the NN
# print('Starting Test', test_dir)
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
#         100 * correct / total))
#
