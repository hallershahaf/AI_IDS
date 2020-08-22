def img2mat(sing_or_mult, bin_or_txt):
    """Imports"""
    import re
    import numpy as np
    import os
    
    """Globals"""
    MTU = 1514
    cols = 32
    rows = int(np.ceil(MTU / cols))
    
    """find all packet files"""
    all_files = os.listdir()
    pack_files = []
    for f in range(len(all_files)):
        if all_files[f].startswith("packet_"):
            pack_files.append(all_files[f]) #pack_files contains the names of all the packet files
    images = len(pack_files) 
    """"create variables from the files"""
    #single file mode
    if sing_or_mult.lower() == "s":
        #textual file mode
        if bin_or_txt.lower() == "t":
            with open(str(pack_files[0]), 'rb') as parsed:
                raw_data = str(parsed.read())
            raw_data = raw_data.split('\n\n')
        #binary file mode
        elif bin_or_txt.lower() == "b":
            packets = np.load(pack_files[0])
    #multiple files mode
    elif sing_or_mult.lower() == "m":
        packets = np.empty((images,rows,cols))
        #textual file mode
        if bin_or_txt.lower() == "t":
            for p in range(len(pack_files)):
                packets[p] = np.loadtxt(pack_files[p], delimiter=',')
        #binary file mode
        elif bin_or_txt.lower() == "b":
            for p in range(len(pack_files)):
                packets[p] = np.load(pack_files[p])