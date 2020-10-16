def random_input(vect_length, exp_probe):
    import numpy as np
    import random as rand

    """Globals"""
    # P = practice
    # T = test
    mtu = 1514
    cols = 32
    packets = 100
    rows = int(np.ceil(mtu / cols))

    """Create desicion vector"""
    exp_or_safe = np.random.choice([0,1], size = vect_length, p = [1 - exp_probe, exp_probe] )
    """Define output
    ################################################
    Each packet is 2-D matrix.
    Each stream is a 3-D matrix.
    The whole vector is a 4-D matrix.
    Meaning: out_mat = [rows of packet,cols of packet, diff packets, diff streams]
    
    # Note: The output order is output = [out_mat, out_valid]
    """
    out_mat = np.zeros((rows, cols, packets, vect_length))
    out_valid = exp_or_safe

    """Create "exploit" data """
    for s in range(vect_length):
        if exp_or_safe[s] == 1:
            for p in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        out_mat[r,c,p,s]= rand.randrange(0, 255)
            """Create "safe" data """
        elif exp_or_safe[s]:
            for p in range(packets):
                for r in range(rows):
                    for c in range(cols):
                        out_mat[r,c,p,s]= rand.randrange(50, 250)
    #Note the order of the output
    output = [out_mat, out_valid]
    return output

