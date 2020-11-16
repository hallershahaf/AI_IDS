import requests
import subprocess
import os
from scapy import sendrecv
from scapy.all import sr1
import scapy.layers.http


def create_pretrain_data(is_exploit, scapy_tcpdump, file_name):
    """
    Create a single file of 100 http packets.
    input:
    --is_exploit = exploit => if true sending gets requests with user datata. [bool]
    --file_name =  where to save the file. [str]

    Note
    -----
    If running in windows, requires  microlap tcpdump saved at C:\\microlap_tcpdump\\tcpdump.exe
    """

    # Creating the request
    sites = ["http://www.walla.co.il:80", "http://www.ynet.co.il:80", "http://13news.co.il:80",
             "http://www.mako.co.il:80", "http://www.israelhayom.co.il:80", "http://www.n12.co.il:80",
             "http://www.google.com:80", "http://bing.com:80", "http://www.yandex.com:80",
             "http://www.swisscows.com:80", "http://search.creativecommons.org/:80", "http://duckduckgo.com:80"]
    values = {'name': 'AI_IDS project',
              'location': 'Technion',
              'language': 'Python'}
    save_command = ">" + os.path.join(os.getcwd(), file_name)

    # Starting the sniff
    if scapy_tcpdump.lower() == "t":
        if os.name == 'nt':
            sniff_command = "C:\\microlap_tcpdump\\tcpdump.exe"
        else:
            sniff_command = "tcpdump"
    else:
        sniff_command = "C:\\Users\\idant\\anaconda3\\python.exe"
        script_path = "/Dataset/Create_Pretrain/http_requests.py"
    if scapy_tcpdump.lower() == "t":
        if is_exploit:
            proc = subprocess.Popen([sniff_command, "-i", "9", "-XX", "-c", "128", "dst", "port", "80",
                                     "-A", "tcp[((tcp[12:1]&0xf0)>>2):4]=0x504F5354",
                                     save_command], shell=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen([sniff_command, "-i", "9", "-XX", "-c", "128", "dst", "port", "80",
                                     "-A", "tcp[((tcp[12:1]&0xf0)>>2):4]=0x47455420",
                                     save_command], shell=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)

    elif scapy_tcpdump.lower() == "s":
        if is_exploit:
            proc = subprocess.Popen([sniff_command, script_path, "True"], shell=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen([sniff_command, script_path, "False"], shell=True , stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)

        sniff = sendrecv.sniff(filter="port 80", count=20)
    else:
        raise Exception("Choose s for scapy or t for tcpdump")

    # Sending get requests
    if scapy_tcpdump.lower() == "t":
        while proc.poll() is None:
            if is_exploit:
                for s in sites:
                    try:
                        requests.post(s, data=values)
                    except requests.exceptions.RequestException:
                        print("Couldn't connect to: ", s)
            else:
                for s in sites:
                    try:
                        requests.get(s)
                    except requests.exceptions.RequestException:
                        print("Couldn't connect to: ", s)
    # print("Finished creating ", file_name)

    # preparse packets for scapy
    if scapy_tcpdump.lower() == "s":
        chosen_packets = []
        for i in range(len(sniff)):
            if ((str(sniff[i].payload.payload.payload.payload)[2:6]) == "POST") \
                    and is_exploit:
                chosen_packets.append(sniff[i].original.hex())
            elif ((str(sniff[i].payload.payload.payload.payload)[2:5]) == "GET") \
                    and not is_exploit:
                chosen_packets.append(sniff[i].original.hex())
        if len(chosen_packets) == 0:
            create_pretrain_data(is_exploit, scapy_tcpdump, file_name)
        else:
            output_data = "SCAPY\n----\n"
            for i in range(len(chosen_packets)):
                output_data = output_data + str(chosen_packets) + "\n"
                output_data = output_data + "----\n"
            with open(os.path.join(os.getcwd(), file_name), "w") as f:
                f.write(output_data)