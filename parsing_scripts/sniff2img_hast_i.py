import re
import numpy as np

def sniff2img(sniff_file):
    """Reads hex dump of tcpdump and transform to images shaped fo HAST I NN"""
    # Globals
    files = IO()
    # Size of images in pixels
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))

    # reading hex
    # Read sniff
    in_file = files.in_dir + "\\" + sniff_file
    with open(str(in_file), 'rb') as sniff:
        packets = str(sniff.read())
    """
    parsing sniff   
    #taking only the packets data
    #at the end of this block, tmp holds the data
    """

    tmp = []
    parsed = packets.split('0x')

    i = 0
    while i < len(parsed) - 1:
        if re.search('\.', parsed[i]):
            i += 1
            data = ''
            while not (re.search('\.', parsed[i])) and i < len(parsed) - 1:
                parsed[i] = re.sub(r'\\n.*$', '', parsed[i])
                parsed[i] = parsed[i][7:]
                parsed[i] = re.sub(r'[^\w]', '', parsed[i])
                data = data + parsed[i]
                i += 1
            tmp.append(data)
    """
    create images
    #creates a 3d matrix where each matrix is an image of a packet
    at the end of this block, parsed holds the parsed images
    """
    packets = len(tmp)
    # pre-converting status
    print("found ", str(packets), " packets")

    parsed = np.full((rows, cols * packets), 255)
    # i = image
    # r = rows
    # c = cols
    for p in range(packets):
        packet_size = len(tmp[p])
        for r in range(rows):
            if r * cols > packet_size - 1:
                break
            else:
                for c in range(cols):
                    if r * cols + 2 * c > packet_size - 1:
                        break
                    else:
                        parsed[r, c + (p * packets)] = int(str(tmp[p][r * cols + 2 * c:r * cols + 2 * c + 2]), 16)
        # mid-converting status
        print("finished ", str(p + 1), " packets of ", str(packets))
    """
    saves the parsed images to a file
    """
    print("saving")
    f_name = files.out_dir + "\\" + "packet_all"
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
