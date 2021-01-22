"""Reads hex dump of wireshark and transform to images"""


# This turns sniffed packets into an image we can feed into the NN
def sniff2img(sniff_file, sing_or_mult, bin_or_txt, wshark_or_tdump):
    """Imports"""
    import re
    import numpy as np
    """Globals"""
    files = IO()
    # Size of images in pixels
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))

    """reading hex"""
    # Read sniff
    in_file = files.in_dir + "\\" + sniff_file
    with open(str(in_file), 'rb') as sniff:
        packets = str(sniff.read())
    """
    parsing sniff   
    #taking only the packets data
    #at the end of this block, tmp holds the data
    """

    # Wireshark mode
    tmp = []
    parsed = []
    if wshark_or_tdump == "w":
        parsed = packets.split('\\r\\n\\r\\n')

        for i in range(len(parsed)):
            tmp.append(parsed[i][75:])
            tmp[i] = re.sub(r'[^\w]', '', tmp[i])
    # tcpdump mode
    if wshark_or_tdump == "t":
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

    parsed = np.full((packets, rows, cols), 255)
    # i = image
    # r = rows
    # c = cols
    for i in range(packets):
        packet_size = len(tmp[i])
        for r in range(rows):
            if r * cols > packet_size - 1:
                break
            else:
                for c in range(cols):
                    if r * cols + 2 * c > packet_size - 1:
                        break
                    else:
                        parsed[i, r, c] = int(str(tmp[i][r * cols + 2 * c:r * cols + 2 * c + 2]), 16)
        # mid-converting status
        print("finished ", str(i), " packets of ", str(packets))
    """
    saves the parsed images to a file
    """
    print("saving")
    # Multiple files mode
    if sing_or_mult.lower() == "m":
        # Textual files mode
        if bin_or_txt.lower() == "t":
            for i in range(packets):
                f_name = files.out_dir + "\\" + "packet_" + str(i) + ".txt"
                np.savetxt(f_name, parsed[i], delimiter=',')
        # Binary files mode
        elif bin_or_txt.lower() == "b":
            for i in range(packets):
                f_name = files.out_dir + "\\" + "packet " + str(i)
                np.save(f_name, parsed[i])
    # Single file mode
    elif sing_or_mult.lower() == "s":
        f_name = files.out_dir + "\\" + "packet_all"
        # Textual file mode
        if bin_or_txt.lower() == "t":
            # i = image
            # r = rows
            # c = cols
            f_name = f_name + ".txt"
            data = ""
            for i in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        if c < cols - 1:
                            data = data + str(parsed[i][r][c]) + ","
                        else:
                            data = data + str(parsed[i][r][c])
                    data = data + "\n"
                data = data + "\n"
            f = open(f_name, 'wb')
            f.write(data)
            f.close()
        # Binary file mode
        elif bin_or_txt.lower() == "b":
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
