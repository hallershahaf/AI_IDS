def random_input(vect_length, exp_probe):
    import numpy as np
    import os

    """Globals"""
    # P = practice
    # T = test
    mtu = 1514
    cols = 32
    packets = 100
    rows = int(np.ceil(mtu / cols))

    """Create decision vector"""
    exp_or_safe = np.random.choice([0, 1], size = vect_length, p = [1 - exp_probe, exp_probe] )
    """Define output
    ################################################
    Each packet is 2-D matrix.
    Each stream is a 3-D matrix.
    The whole vector is a 4-D matrix.
    Meaning: out_mat = [rows of packet,cols of packet, diff packets, diff streams]
    
    # Note: The output order is output = [out_mat, out_valid]
    """
    out_mat = np.zeros((vect_length, packets, rows, cols))
    out_valid = exp_or_safe

    """Create "exploit" data """
    for s in range(len(exp_or_safe)):
        if exp_or_safe[s] == 1:
            for p in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        out_mat[s,p,r,c]= np.random.randint(0, 255)
    #Create "safe" data
        elif exp_or_safe[s] == 0:
            for p in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        out_mat[s,p,r,c]= np.random.randint(50, 250)
    #Note the order of the output
    output = [out_mat, out_valid]
    # Saves the output to files for the dataset
    os.makedirs("Dataset")
    for s in range(len(exp_or_safe)):
        np.save(os.path.join("Dataset", str(s)), out_mat[s])
    np.save(os.path.join("Dataset", "eOs"), out_valid)
    return output

