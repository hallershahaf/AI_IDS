import os


def rdp_connect(ip_addr):
    commnd = "mstsc /control /admin /V:" + str(ip_addr)
    os.system(commnd)
