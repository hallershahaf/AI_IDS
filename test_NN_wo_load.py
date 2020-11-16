# Function for testing an NN
import os
# self made functions and classes
import AI_IDS.create_dataset_v2 as dataset
from AI_IDS.HAST_I_NN import HAST_I

# Pytorch imports
import torch


def test_wo_load(dir_name, state_dict):

    # Define Data for the test
    test_dir = os.path.join(os.getcwd()[:-6], dir_name)
    test_set = dataset.create_dataset(test_dir, "EoS.npy")
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    # Load the network
    net = HAST_I()
    net.load_state_dict(state_dict)

    # Run the test
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # print(images.size())
            # print(type(images))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ', str(len(os.listdir(test_dir)) - 1), 'test images: %d %%' % (
            100 * correct / total))

    return (100 * correct) / total