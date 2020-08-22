"""Reads hex dump of wireshark and trnsform to images"""
def sniff2img(sniff_file, sing_or_mult, bin_or_txt, wshark_or_tdump):
    """Imports"""
    import re
    import numpy as np
    
    """Globals"""
    #size of images in pixels
    MTU = 1514
    cols = 32
    rows = int(np.ceil(MTU / cols))

    """reading hex"""
    #Read sniff
    with open(sniff_file, 'rb') as sniff:
        packets = str(sniff.read())
    """
    parsing sniff   
    #taking only the packes data
    #at the end of this block, tmp holds the data
    """
    #wireshark mode
    if wshark_or_tdump = "w":
        parsed = packets.split('\\r\\n\\r\\n')
        tmp = []

        for i in range(len(parsed)):
            tmp.append(parsed[i][75:])
            tmp[i] = re.sub(r'[^\w]','',tmp[i])

    """
    create images
    #creates a 3d matrix where each matrix is an image of a packet
    at the end of this block, parsed holds the parsed images
    """
    packets = len(tmp)
    parsed = np.full((packets,rows,cols),255)

    #i = image
    #r = rows
    #c = cols
    for i in range(packets):
        packet_size =len(tmp[i])
        for r in range(rows):
            if (r * cols > packet_size -1):
                break
            else:
                for c in range(cols):
                    if (r * cols + 2 * c > packet_size -1):
                        break
                    else:
                        parsed[i, r, c] = int(str(tmp[i][r * cols + 2 * c :r * cols + 2 * c + 2]),16)
    """
    saves the parsed images to a file
    """
    #multiple files mode
    if sing_or_mult.lower() == "m":
        #textual files mode
        if bin_or_txt.lower() == "t":
            for i in range(packets):
                f_name = "packet_" + str(i) + ".txt"
                np.savetxt(str(f_name), parsed[i], delimiter = ',')
        #binary files mode
        elif bin_or_txt.lower() == "b":
            for i in range(packets):
                f_name = "packet " + str(i)
                np.save(str(f_name), parsed[i])
    #single file mode
    elif sing_or_mult.lower() == "s":
        f_name = "packet_all"
        #textual file mode
        if bin_or_txt.lower() == "t":
            #i = image
            #r = rows
            #c = cols
            f_name = f_name + ".txt"
            data = ""
            for i in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        if c < cols - 1:
                            data = data + str(parsed[i][r][c]) +","
                        else:
                            data = data + str(parsed[i][r][c])
                    data = data + "\n"
                data = data + "\n"
            f = open(str(f_name), 'wb')
            f.write(data)
            f.close()
        #binary file mode
        elif bin_or_txt.lower() == "b":
            np.save(str(f_name), parsed)