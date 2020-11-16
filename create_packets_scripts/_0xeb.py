import subprocess
import os
import time

# Process handler
_0xeb = None

# Command line
_0xeb_cmd = ["python3", "/home/shahaf/bluekeep/0xeb-bp/win7_32_poc.py"]

# End line
end_line = "allocating fake objects"

def run0xeb():
	devnull = open(os.devnull, 'w')
	global _0xeb
	_0xeb = subprocess.run(_0xeb_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=devnull)

