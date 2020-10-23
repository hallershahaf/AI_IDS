import os
# TODO make the code suitable for ubuntu terminal rdp to windows (maybe via metasploit)


def rdp_connect(ip_addr):
    cmd = "mstsc /control /admin /V:" + str(ip_addr)
    os.system(cmd)
