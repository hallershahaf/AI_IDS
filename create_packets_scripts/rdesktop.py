import os
import signal
import subprocess
import time

delay_setup = 2

cmd_line = ["sudo", "rdesktop", "192.168.1.42", "-u", "Shahaf", "-p", "AIids123456"]

def runRDesktop():
	devnull = open(os.devnull, "w")
	rdesktop = subprocess.Popen(cmd_line, stdin=subprocess.PIPE, stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
	time.sleep(delay_setup)
	while rdesktop.poll() is None:
		rdesktop.terminate()
	time.sleep(delay_setup)
