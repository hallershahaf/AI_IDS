# Run HAST I with noises + creating a graph for datatest accuracy for each epoch
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

# Defining the data for the NN
batch_size = 64
train_dir = os.path.join(os.getcwd(), "..", "Dataset_HAST_I_0_big")  # The location of the Dataset folder
train_set = dataset.create_dataset(train_dir, "EoS.npy")  # Creating the dataset
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('safe', 'exploit')  # 0 = safe, 1 = exploit

# Defining the NN
# The lr, momentum and transform are based on pytorch example
net = HAST_I()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# NN Variables definition
epochs = 2
packets = 128
mtu = 1514
cols = 32
rows = int(np.ceil(mtu / cols))
extra_packets = 3
samples = len(os.listdir(train_dir))
batch_per_epoch = np.ceil(samples / batch_size)
accuracy_check_step = 4

# List of accuracy results for different test sets and through the training.
# Used for the print statistics part at the end
Datatest_100_accuracy = []
Datatest_75_accuracy = []
Datatest_50_accuracy = []
Datatest_25_accuracy = []
Datatrain_accuracy = []

# Training the NN
for epoch in range(epochs):  # loop over the dataset multiple times
    print("epoch = ", str(epoch))
    print("lr = ", scheduler.get_last_lr()[0])
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # Move the stream at random amount of packets. The indentation of the stream is in the range [0, 3]
        indent = np.random.randint(0, extra_packets + 1)
        cropped_inputs = inputs[:, :, :, (cols * (indent + 1)):((packets * cols) + (cols * (indent + 1)))]

        # forward + backward + optimize
        outputs = net(cropped_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate statistics
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        running_loss += loss.item()
        # print statistics after each batch of samples

        if i % accuracy_check_step == accuracy_check_step - 1:  # print every accuracy_check_step mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 4))
            accuracy_train = 100.0 * correct_train / total_train
            Datatrain_accuracy.append(accuracy_train)
            print('accuracy = %.2f %%' % accuracy_train)
            running_loss = 0.0

    # Validation after each epoch
    net.eval()
    with torch.no_grad():
        Datatest_100_accuracy.append(twol("Datatest_HAST_I_0", net.state_dict()))
        Datatest_75_accuracy.append(twol("Datatest_HAST_I_diff_safe&safe", net.state_dict()))
        Datatest_50_accuracy.append(twol("Datatest_HAST_I_0xeb&safe", net.state_dict()))
        Datatest_25_accuracy.append(twol("Datatest_HAST_I_0xeb&safe&diff_safe", net.state_dict()))
    net.train()

    # update lr
    scheduler.step()

print('Finished Training')

# Saving NN state
print('Saving state')
torch.save({
    'epoch': epochs,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, os.path.join(os.getcwd(), "metasploit_post-training_NN", "NN_post_training_HAST_I_6_e_V2"))

# Printing the Statistics graph
print("Printings statistics")
# Defining the plots
train_accuracy_precision = accuracy_check_step / batch_per_epoch
print(train_accuracy_precision)
print(len(Datatrain_accuracy))
print(len(Datatrain_accuracy) / batch_per_epoch)
plt.plot(np.arange(1, len(Datatest_100_accuracy) + 1), Datatest_100_accuracy, label="100% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_75_accuracy) + 1), Datatest_75_accuracy, label="75% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_50_accuracy) + 1), Datatest_50_accuracy, label="50% similar", marker='o')
plt.plot(np.arange(1, len(Datatest_25_accuracy) + 1), Datatest_25_accuracy, label="25% similar", marker='o')
# plt.plot(np.arange(0, (batch_per_epoch / Datatrain_accuracy) + train_accuracy_precision, train_accuracy_precision),
#          Datatrain_accuracy, "k--", label="Datatrain accuracy")

# Defining the grids
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(np.arange(1, epochs + 1, 1))
# Creating labels and titles
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Vs. Epochs - SGD')
plt.legend()
# Saving and showing the plot
plot_name = "Accuracy Vs epochs for" + str(epochs) + " SGD.png"
plt.savefig(os.path.join(os.getcwd(), plot_name))
# plt.show()
