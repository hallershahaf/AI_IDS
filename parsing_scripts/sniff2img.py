"""Reads hex dump of wireshark and transform to images"""


def sniff2img(sniff_file, sing_or_mult, bin_or_txt):
    """Imports"""
    import re
    import numpy as np
    """Globals"""
    # Size of images in pixels
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))

    """reading hex"""
    # Read sniff
    with open(str(sniff_file), 'rb') as sniff:
        packets = str(sniff.read())

    """
    parsing sniff   
    #taking only the packets data
    #at the end of this block, tmp holds the data
    """
    parsed = packets.split('\\r\\n\\r\\n')
    tmp = []

    for i in range(len(parsed)):
        tmp.append(parsed[i][75:])
        tmp[i] = re.sub(r'[^\w]', '', tmp[i])

    """
    create images
    #creates a 3d matrix where each matrix is an image of a packet
    at the end of this block, parsed holds the parsed images
    """
    packets = len(tmp)
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
    """
    saves the parsed images to a file
    """
    # Multiple files mode
    if sing_or_mult.lower() == "m":
        # Textual files mode
        if bin_or_txt.lower() == "t":
            for i in range(packets):
                f_name = "packet_" + str(i) + ".txt"
                np.savetxt(str(f_name), parsed[i], delimiter=',')
        # Binary files mode
        elif bin_or_txt.lower() == "b":
            for i in range(packets):
                f_name = "packet " + str(i)
                np.save(str(f_name), parsed[i])
    # Single file mode
    elif sing_or_mult.lower() == "s":
        f_name = "parsed_sniff"
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
            f = open(str(f_name), 'wb')
            f.write(data)
            f.close()
        # Binary file mode
        elif bin_or_txt.lower() == "b":
            np.save(str(f_name), parsed)
