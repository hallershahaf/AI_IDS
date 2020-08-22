def img2mat(sing_or_mult, bin_or_txt):
    """Imports"""
    import numpy as np
    import os
    
    """Globals"""
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))
    """find all packet files"""
    all_files = os.listdir()
    pack_files = []
    packets = []
    for f in range(len(all_files)):
        if all_files[f].startswith("packet_"):
            # pack_files contains the names of all the packet files
            pack_files.append(all_files[f])
    images = len(pack_files)
    # pre-parsing status
    print("found ", str(images), " packets")
    """"create variables from the files"""
    # Single file mode
    if sing_or_mult.lower() == "s":
        # Only Binary file mode
        packets = np.load(pack_files[0])
    # Multiple files mode
    elif sing_or_mult.lower() == "m":
        packets = np.empty((images, rows, cols))
        # Textual file mode
        if bin_or_txt.lower() == "t":
            for p in range(len(pack_files)):
                packets[p] = np.loadtxt(pack_files[p], delimiter=',')
                # mid-parsing status
                print("finished ", str(p+1), " of ", str(images))
        # Binary file mode
        elif bin_or_txt.lower() == "b":
            for p in range(len(pack_files)):
                packets[p] = np.load(pack_files[p])
                # mid-parsing status
                print("finished ", str(p + 1), " of ", str(images))
    return packets
