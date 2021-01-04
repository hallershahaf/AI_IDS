# Find the first packets after the last packet containing hex string: [14 03 ?? 00 01 01 16]
import numpy as np


def first_encrypted(stream, packets):
    """
    Stream = np.array of size [packets, rows, cols]
    packets = amount of packets
    output = The first encrypted packet
    """
    hex_tls_handshake = ['14', '?', '?', '00', '01', '01', '16']
    int_tls_handshake = []
    # Transform TLS handshake from hex to int
    for hex_val in hex_tls_handshake:
        if hex_val != '?':
            int_tls_handshake.append(int(hex_val, 16))
        else:
            int_tls_handshake.append('?')

    # Find last occurrence of TLS handshake
    rows = stream.shape[1]
    cols = int(stream.shape[2] / packets)
    max_packet = 0
    for p in range(packets):
        for r in range(rows):
            for c in range(cols):
                if stream[0, r, c + (p * cols)] == int_tls_handshake[0]:
                    cells_checked = 1
                    c_shift = 1
                    r_shift = 0
                    # While haven't Reached the end of the packet or reached full match
                    while (cells_checked < len(int_tls_handshake) and
                           r_shift < rows and c + (p * cols) + c_shift < stream.shape[2]):
                        # Reached the end of the packet's columns
                        if c + p * cols + c_shift == p * (cols + 1):
                            c_shift = 0
                            r_shift += 1
                            continue
                        # A joker
                        elif int_tls_handshake[cells_checked] == '?':
                            c_shift += 1
                            cells_checked += 1
                            continue
                        # Got to the end of a packets
                        elif r + r_shift >= rows:
                            break
                        # Check if suits the pattern
                        else:
                            # Fits the pattern
                            if stream[0, r + r_shift, c + (p * cols) + c_shift] == int_tls_handshake[cells_checked]:
                                cells_checked += 1
                                c_shift += 1
                            # Doesn't fits the pattern
                            else:
                                break
                    # Found a new max_packet
                    if cells_checked == len(int_tls_handshake):
                        # print(p)
                        max_packet = p
    return max_packet + 1
