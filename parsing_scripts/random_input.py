def random_input(exp_safe, vect_length):
    import numpy as np
    import random as rand

    """Globals"""
    # P = practice
    # T = test
    mtu = 1514
    cols = 32
    rows = int(np.ceil(mtu / cols))
    packets = vect_length;

    """Define output"""
    out_vect  = np.full((packets, rows, cols), 255)
    out_valid = np.full((packets, 1, 1), 255)

    """Create "exploit" data """
    if exp_safe == "e":
        for p in range(packets):
            for i in range(rows):
                for j in range(cols):
                    out_vect[p,i,j]= rand.randrange(0, 255)
            out_valid[p] = '1'
        """Create "safe" data """
    elif exp_safe == "s":
        for p in range(packets):
            for i in range(rows -1):
                for j in range(cols -1 ):
                    out_vect[p,i,j]= rand.randrange(50, 250)
            out_valid[p] = '0'

    output = [out_vect, out_valid]
    return output

