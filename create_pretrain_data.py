import requests
import subprocess
import os


def create_pretrain_data(is_exploit, file_name):
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
    if os.name == 'nt':
        proc = subprocess.Popen(["C:\\microlap_tcpdump\\tcpdump.exe", "-i", "9", "-XX", "-c", "100", "port", "80",
                                 save_command], shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(["tcpdump", "-i", "9", "-XX", "-c", "100", "port", "80",
                                 save_command], shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    # Sending get requests
    while proc.poll() is None:
        if is_exploit:
            for s in sites:
                requests.get(s, values)
        else:
            for s in sites:
                requests.get(s)

    # print("Finished creating ", file_name)
