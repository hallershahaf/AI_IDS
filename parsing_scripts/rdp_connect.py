import os
#TODO make the code suituble for ubuntu terminal rdp to windows (maybe via metasploit)

def rdp_connect(ip_addr):
    commnd = "mstsc /control /admin /V:" + str(ip_addr)
    os.system(commnd)
