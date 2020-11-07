# Load and test a NN
import os
# self made functions and classes
import AI_IDS.create_dataset_v2 as dataset
from AI_IDS.HAST_I_NN import HAST_I

# Pytorch imports
import torch

# Define Data for the test
test_dir = os.path.join(os.getcwd()[:-6], "Datatest_HAST_I_0xeb&safe")
test_set = dataset.create_dataset(test_dir, "EoS.npy")
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

# Load the network
classes = ('safe', 'exploit')
net = HAST_I()
# Loaded_net = torch.load(".\\metasploit_post-training_NN\\NN_post_training_HAST_I_5_e")
Loaded_net = torch.load(".\\metasploit_post-training_NN\\NN_post_training_HAST_I_5_e")
net.load_state_dict(Loaded_net['model_state_dict'])
# net.load_state_dict(Loaded_net)

# Run the test
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
