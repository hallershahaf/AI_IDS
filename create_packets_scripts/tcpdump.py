import subprocess
import signal
import time
import os

tcpdump_process = None

# The system's interface to the outside world
interface = "ens33"
# Port 3389 is reserved for RDP communication
port = 3389
# Number of packets before we close tcpdump
packet_num = 200
# The command with all the flags
tcpdump_cmd = ["sudo", "tcpdump", "-i", interface, "-XX", "-c", str(packet_num), "port" , str(port)]


def runTcpdump(output_file):
	global tcpdump_process
	output = open(output_file, "w")
	devnull = open(os.devnull, "w")
	tcpdump_process = subprocess.Popen(tcpdump_cmd, stdout=output, stderr=devnull)

def stopTcpdump():
	tcpdump_process.send_signal(signal.SIGINT)
