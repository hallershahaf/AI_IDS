import requests
import subprocess

#TODO make sure that you check os type and that you choose if you want to create exploit or not + the you can choose the parameters for the tcp_dump
def create_pretrain():
    sites = ["http://www.walla.co.il", "http://www.ynet.co.il", "http://www.google.com"]
    values = {'name' : 'Michael Foord',
              'location' : 'Northampton',
              'language' : 'Python' }

    X = subprocess.Popen(["C:\microlap_tcpdump\\tcpdump.exe", "-i", "9", "-XX", "-c", "100", "port", "80", ">C:\\Users\idant\Desktop\\test2"],shell=True ,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # subprocess.run("C:\microlap_tcpdump\\tcpdump.exe -i 9 -XX -c 100 port 80 > C:\\Users\idant\Desktop\\test2", stdout=subprocess.DEVNULL)
    i = 0
    while i < 30:
        i += 1
        for s in sites:
            requests.get(s)