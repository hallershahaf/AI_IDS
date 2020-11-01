import numpy as np
from AI_IDS.create_pretrain_data import create_pretrain_data as cpd
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img as s2i
import os
import shutil
import datetime


def create_pretrain_dataset(stream_num, ratio, folder_name):
    """
    Create dateset to pretrain the NN where:
    *HTTP/get = safe
    *HTTP/get + values = exploit

    Input:
    Stream_num = amount of streams to create [int]
    ratio = the part of exploits of the streams [float between 0-1]
    folder_name = where to save the streams
    """
    # Check how much time the process takes
    start_time = datetime.datetime.now()

    # Create the output folder
    if os.path.exists(folder_name):
        delete = str(input("Are you sure you want to delete " + folder_name + ": [y/n]"))
        if delete.lower() == 'y':
            shutil.rmtree(folder_name)
        else:
            exit()
    os.makedirs(folder_name)
    dest_dir = os.path.join(os.getcwd(), folder_name)

    # Sniffing the data
    eos = np.random.choice([0, 1], size=stream_num, p=[1 - ratio, ratio])
    np.save(os.path.join(dest_dir, "EoS"), eos)
    file_names = []
    print("Sniffing")
    for s in range(stream_num):
        file_name = os.path.join(dest_dir, str(s) + ".txt")
        if eos[s] == 0:
            cpd(False, file_name)
        else:
            cpd(True, file_name)
        # print(file_name)
        file_names.append(file_name)
        s2i(file_name, file_name[:-4])
        print(str(s+1), " of ", str(stream_num))

    # # Converting the sniffs to images
    # print("\nConverting to images")
    # for s in range(stream_num):
    #     s2i(file_names[s], file_names[s][:-4])
    #     print(str(s + 1), " of ", str(stream_num))
    #
    # # Save the EoS file
    # np.save(os.path.join(dest_dir, "EoS"), eos)

    # Calculates the time delta
    finish_time = datetime.datetime.now()
    time_delta = finish_time - start_time
    print("The process took ", str(time_delta))