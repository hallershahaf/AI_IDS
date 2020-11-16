import os
import subprocess
import time

delay_setup = 10

def runRemmina():
	devnull = open(os.devnull, "w")
	remmina = subprocess.Popen(["remmina"], stdin=subprocess.PIPE, stdout=devnull, stderr=devnull)
	time.sleep(delay_setup)
	remmina.terminate()
	
