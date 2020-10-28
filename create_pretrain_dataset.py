import numpy as np
from AI_IDS.create_pretrain_data import create_pretrain_data as cpd
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img as s2i
import os
import shutil


def create_pretrain_dataset(stream_num, ratio, folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    dest_dir = os.path.join(os.getcwd(), folder_name)

    eos = np.random.choice([0, 1], size=stream_num, p=[1 - ratio, ratio])
    file_names = []
    print("Sniffing")
    for s in range(stream_num):
        file_name = os.path.join(dest_dir, str(s) + ".txt")
        if eos[s] == 0:
            cpd(False, file_name)
        else:
            cpd(True, file_name)
        file_names.append(file_name)
        print(str(s+1), " of ", str(stream_num))
    print("\nConverting to images")
    for s in range(stream_num):
        s2i(file_names[s], file_names[s][:-4])
        print(str(s + 1), " of ", str(stream_num))
    np.save(os.path.join(dest_dir, "EoS"), eos)
