# Transform every plain text packet to 0

import numpy as np
import os
import AI_IDS.parsing_scripts.first_encrypted as fe


def delete_plain_text(source_fold, dest_fold, packets):
    """
    source_fold = location of source np.array-s
    dest_fold = where to keep the output
    packets = amount of packets
    output = The first encrypted packet
    """
    source_dir = os.listdir(source_fold)
    streams = len(source_dir)
    i = 0
    for file in source_dir:
        # Output file
        f_name = file[0:-4]
        f_name = os.path.join(dest_fold, f_name)

        # Find the last plain text packet and zero all up-to that packet
        cur_stream = np.load(os.path.join(source_fold, file))
        cols = int(cur_stream.shape[2] / packets)
        first_encrypted = fe.first_encrypted(cur_stream, packets)
        cur_stream[:, :, 0:((first_encrypted - 1) * cols)] = 0

        # Save output
        np.save(f_name ,cur_stream)
        i += 1
        print("Finished " + str(i) + " from " + str(streams))
    print("Finished deleting plain text packets")
