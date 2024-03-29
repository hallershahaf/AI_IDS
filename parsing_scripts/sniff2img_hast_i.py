import re
import numpy as np
# import os


def sniff2img(sniff_file, out_file, stream_length, shift_stream, packets2move):
    """Reads hex dump of tcpdump and transform to images shaped fo HAST I NN"""
    # Globals
    # files = IO()
    # Size of images in pixels
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))

    parsed = []

    # reading hex
    # Read sniff
    # in_file = files.in_dir + "\\" + sniff_file
    in_file = sniff_file
    with open(str(in_file), 'r', errors='ignore') as sniff:
        packets = str(sniff.read())
    """
    parsing sniff   
    #taking only the packets data
    #at the end of this block, tmp holds the data
    """
    # Check if the file was made by scapy or not
    if bool(re.search("SCAPY\n", packets)):
        is_scapy = True
    else:
        is_scapy = False

    if not is_scapy:
        p_bytes = packets.split("\n")
        i = 0
        p = 0
        while i < len(p_bytes) - 1:
            if bool(re.search('\d IP ', p_bytes[i])):
                i += 1
                p += 1
                data = ""
                while not bool(re.search('\d IP ', p_bytes[i])) and i <= len(p_bytes) - 1 and len(p_bytes[i]) > 1:
                    tmp = re.sub("^.*0x.*:  ", '', p_bytes[i])
                    try:
                        tmp = tmp[0:re.search('  .*$', tmp).start()]
                    except:
                        tmp = tmp
                    tmp = re.sub(r'[^\w]', '', tmp)
                    data = data + tmp
                    if i == len(p_bytes) - 1:
                        break
                    else:
                        i += 1
                parsed.append(data)
        # print("Found ", str(p), " packets")

    elif is_scapy:
        p_bytes = packets.split("\n")
        p_bytes = p_bytes[2:]
        for p in p_bytes:
            if bool(re.search("----", p)) or len(p) < 2:
                continue
            else:
                tmp = re.sub(r'[^\w]', '', p)
                parsed.append(tmp)
    """
    create images
    #creates a 3d matrix where each matrix is an image of a packet
    at the end of this block, parsed holds the parsed images
    """
    if shift_stream.lower() == "l":
        start_i = packets2move
        last_i = stream_length + packets2move
        movement = -1
    elif shift_stream.lower() == "r":
        start_i = 0
        last_i = stream_length - packets2move
        movement = 1
    else:
        start_i = 0
        last_i = stream_length
        movement = 0

    tmp = parsed
    packets = len(tmp)
    # pre-converting status
    # print("found ", str(packets), " packets")
    if packets < stream_length:
        print("Found ", str(packets), " packets")
        exit()
    # parse only 128 packets
    depth = stream_length

    parsed = np.zeros((1, rows, cols * depth))
    # i = image
    # r = rows
    # c = cols
    for p in range(start_i, last_i):
        packet_size = len(tmp[p])
        for r in range(rows):
            if r * cols > packet_size - 1:
                break
            else:
                for c in range(cols):
                    if r * cols + 2 * c > packet_size - 1:
                        break
                    else:
                        parsed[0, r, c + ((p + movement * packets2move) * cols)] = \
                            int(str(tmp[p][r * cols + 2 * c:r * cols + 2 * c + 2]), 16)
        # mid-converting status
        # print("finished ", str(p + 1), " packets of ", str(packets))
    """
    saves the parsed images to a file
    """
    # print("saving ", out_file)
    # f_name = os.path.join(files.out_dir, out_file)
    f_name = out_file
    np.save(f_name, parsed)


""" Defines a class of the input and output source"""


class IO:
    def __init__(self):
        import os
        self.in_dir = os.path.abspath(os.getcwd())
        self.out_dir = os.path.abspath(os.getcwd())

    def in_dir(self):
        directory = str(input('Please write the input directory in a proper python manner: \n'))
        self.in_dir = directory

    def out_dir(self):
        directory = str(input('Please write the output directory in a proper python manner: \n'))
        self.out_dir = directory
