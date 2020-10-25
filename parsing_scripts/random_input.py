import numpy as np
import numpy.random as nrand
import os
import shutil


def random_input(num_of_classes, vector_length, depth, rows, cols):

    print("\nCreating", vector_length, "random packets of size [", depth, ",", rows, ", ", cols,
          "] , between", num_of_classes, "different classes:")

    # Create decision vector
    class_labels = np.random.choice(list(range(0, num_of_classes)), size=vector_length)

    # Define output
    ################################################
    # Each packet is 2-D matrix.
    # Each stream is a 3-D matrix.
    # The whole vector is a 4-D matrix.
    # Meaning: out_mat = [rows of packet,cols of packet, diff packets, diff streams]
    
    # Note: The output order is output = [out_mat, out_valid]
    out_mat = np.zeros((vector_length, depth, rows, cols))
    out_valid = class_labels

    # Create random data
    for s in range(len(class_labels)):
        # Print progress
        if (s + 1) % 10 == 0:
            print("Current progress -> ", s + 1)
        out_mat[s] = np.ones((depth, rows, cols)) * class_labels[s]

    # Note the order of the output
    output = [out_mat, out_valid]
    # Saves the output to files for the dataset
    if os.path.exists("Dataset"):
        shutil.rmtree("Dataset")
    os.makedirs("Dataset")
    for s in range(len(class_labels)):
        np.save(os.path.join("Dataset", str(s)), out_mat[s])
    np.save(os.path.join("Dataset", "EoS"), out_valid)
    print("Done creating random input...")
