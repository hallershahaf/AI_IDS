# Run HAST I with noises + creating a graph for datatest accuracy for each epoch
import os
from datetime import datetime as time
# self made functions and classes
import AI_IDS.create_dataset_v2 as dataset
from AI_IDS.HAST_I_NN import HAST_I
from AI_IDS.test_NN_wo_load import test_wo_load as testNet

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np

# Normalization stuffs
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Measure time so we can compare different batch and hardware setups
start = time.now()

# Defining the datasets for the NN
train_dir_50_25_25 = os.path.join(os.getcwd(), "..\\Datatrain\\Train_50_25_25")  # The location of the Dataset folder
train_set_50_25_25 = dataset.create_dataset(train_dir_50_25_25, "EoS.npy")  # Creating the dataset
train_dir_50_50 = os.path.join(os.getcwd(), "..\\Datatrain\\Train_50_50")  # The location of the Dataset folder
train_set_50_50 = dataset.create_dataset(train_dir_50_50, "EoS.npy")  # Creating the dataset
train_sets = [train_set_50_50, train_set_50_25_25]

# classes = ('safe', 'exploit')  # 0 = safe, 1 = exploit

# Send to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining the NN

# NN Variables definition
###############################################
# Above 16 starts to bottleneck the GPU VRAM
batch_size = 1
# This is to try and avoid overfitting
noise_values = [0, 1, 3, 5]
epochs = 10

# We only sniffed 200 packets, so above is currently not supported
packets = 200

# Generic image definitions
mtu = 1514
cols = 32
rows = int(np.ceil(mtu / cols))
samples = len(os.listdir(train_dir_50_50))
batch_per_epoch = np.ceil(samples / batch_size)

# NN loops aid variables
accuracy_check_step = 500
max_reattempts = 2
reattempts = 0
valid_output = False
warning_flag = False
bad_configs = []
###############################################

# List of accuracy results for different test sets.
# Used to create statistics at the end of a run
Datatest_100_accuracy = []
Datatest_75_accuracy = []
Datatest_50_accuracy = []
Datatest_25_accuracy = []
Datatrain_accuracy = []

current_config_string = ""
opt_string = ""

# We create a dummy net for the parameters.
net = HAST_I(packets)
criterion = nn.CrossEntropyLoss()

for current_noise in noise_values:
    noise_string = str(current_noise)
    for train_set in train_sets:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        set_string = "50_25_25" if train_set == train_set_50_25_25 else "50_50"
        for optimizer in [0, 1]:
            valid_output = False
            reattempts = 0
            while reattempts < max_reattempts and not valid_output:
                # We need to reset the net between runs
                # So delete to clear from GPU and send again
                del net
                net = HAST_I(packets).to(device)

                if optimizer == 0:
                    opt = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=0)
                    opt_string = "RMS"
                else:
                    opt = optim.SGD(net.parameters(), lr=0.001, weight_decay=0)
                    opt_string = "SGD"

                current_config_string = set_string + " " + opt_string + " " + noise_string
                print("Running " + current_config_string)

                scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

                for epoch in range(epochs):  # loop over the dataset multiple times
                    before = time.now()
                    running_loss = 0.0
                    correct_train = 0
                    total_train = 0
                    for i, data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data

                        # zero the parameter gradients
                        opt.zero_grad()

                        # Move the stream at random amount of packets.
                        indent = np.random.randint(0, current_noise + 1)

                        cropped_inputs = (inputs[:, :, :, (cols * (indent + 1)):((packets * cols)
                                                                                 + (cols * (indent + 1)))]).to(device)

                        # forward + backward + optimize
                        # NOTE - outputs will be in GPU without implicit transfer from us
                        outputs = net(cropped_inputs, packets)

                        # GPU has limited memory, we need to clear as soon as we can
                        del cropped_inputs
                        torch.cuda.empty_cache()

                        # Only labels is not in GPU yet, lets YEET it over
                        labels = labels.to(device)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        opt.step()

                        # calculate statistics
                        _, predicted_train = torch.max(outputs.data, 1)
                        total_train += labels.size(0)
                        correct_train += (predicted_train == labels).sum().item()
                        running_loss += loss.item()
                        # print statistics after each batch of samples

                        # GPU has limited memory, we need to clear as soon as we can
                        del labels
                        del outputs
                        torch.cuda.empty_cache()

                        # print every accuracy_check_step mini-batches
                        if i % accuracy_check_step == accuracy_check_step - 1:
                            print('[%d, %5d] loss: %.3f' %
                                  (epoch + 1, i + 1, running_loss / 4))
                            accuracy_train = 100.0 * correct_train / total_train
                            Datatrain_accuracy.append(accuracy_train)
                            print('accuracy = %.2f %%' % accuracy_train)
                            running_loss = 0.0

                    # Validation after each epoch
                    net.eval()
                    with torch.no_grad():
                        Datatest_100_accuracy.append(testNet(".\\Datatest\\Test100", net, packets))
                        Datatest_75_accuracy.append(testNet(".\\Datatest\\Test75", net, packets))
                        Datatest_50_accuracy.append(testNet(".\\Datatest\\Test50", net, packets))
                        Datatest_25_accuracy.append(testNet(".\\Datatest\\Test25", net, packets))
                    net.train()

                    # update lr
                    scheduler.step()

                    after = time.now()
                    print("Time for epoch : ", after - before)

                # Check if the output is valid
                if max(Datatest_100_accuracy) < 75 and max(Datatest_75_accuracy) < 75:
                    reattempts += 1
                else:
                    valid_output = True

            if not valid_output:
                warning_flag = True
                bad_configs = bad_configs + [current_config_string]

            print('Finished Training current configuration')

            # Printing the Statistics graph
            print("Printings statistics")
            # Defining the plots
            train_accuracy_precision = accuracy_check_step / batch_per_epoch
            plt.plot(np.arange(1, len(Datatest_100_accuracy) + 1), Datatest_100_accuracy,
                     label="50% old exp, 50% old RDP", marker='o')
            plt.plot(np.arange(1, len(Datatest_75_accuracy) + 1), Datatest_75_accuracy,
                     label="50% old exp, 25% old/new RDP", marker='o')
            plt.plot(np.arange(1, len(Datatest_50_accuracy) + 1), Datatest_50_accuracy,
                     label="50% new exp, 50% old RDP", marker='o', linestyle='dashdot')
            plt.plot(np.arange(1, len(Datatest_25_accuracy) + 1), Datatest_25_accuracy,
                     label="50% new exp, 25% old/new RDP", marker='o', linestyle='dotted')

            # plt.plot(np.arange(train_accuracy_precision, (len(Datatrain_accuracy) * train_accuracy_precision) +
            #                    train_accuracy_precision, step=train_accuracy_precision), Datatrain_accuracy, "k--",
            #          label="Datatrain accuracy")
            # Defining the grids

            plot_name = opt_string + "_" + noise_string + "noise_" + set_string
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.xticks(np.arange(1, epochs + 1, 1))
            # Creating labels and titles
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.title("Encryption only - " + plot_name)
            plt.legend()
            plt.ylim(0, 100)
            # Saving and showing the plot
            plt.savefig(os.path.join(os.getcwd() + "/../Graphs", plot_name))
            plt.close()
            # Showing the plt stops the code until exiting the graph
            # plt.show()

end = time.now()
print("Total running time is", end - start)

if warning_flag:
    print("Some configurations had bad outputs:")
    for string in bad_configs:
        print("\t*\t" + string)
