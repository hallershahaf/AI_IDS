import os
import time
from metasploit	 import *
from tcpdump	 import *
from remmina	 import *
from _0xeb	 	 import *
from rdesktop 	 import *


packet_num = 50
delay_exploit = 2

# folder_exploit		= "Data_exploit"
# folder_safe	 	= "Data_safe"
# folder_0xeb	 	= "Data_0xeb"
# folder_rdesktop  	= "Data_rdesktop"
folder_exploit_noscan	= "Data_exploit_noscan" 

# index_file_0xeb 		 = open(folder_0xeb		 + "/index", "r+")
# index_file_safe			 = open(folder_safe		 + "/index", "r+") 
# index_file_exploit		 = open(folder_exploit		 + "/index", "r+")
# index_file_rdesktop		 = open(folder_rdesktop		 + "/index", "r+")
index_file_exploit_noscan	 = open(folder_exploit_noscan	 + "/index", "r+")

print("Reading indexes from index files...")

# index_safe 		 = int(index_file_safe.read())
# index_exploit		 = int(index_file_exploit.read())
# index_0xeb		 = int(index_file_0xeb.read())
# index_rdesktop		 = int(index_file_rdesktop.read())
index_exploit_noscan	 = int(index_file_exploit_noscan.read())

# print("Starting 0xeb		packets from index " + str(index_0xeb))
# print("Starting exploit	packets from index " + str(index_exploit))
# print("Starting safe		packets from index " + str(index_safe))
# print("Starting rdesktop	packets from index " + str(index_rdesktop))
print("Starting noscan_exploit	packets from index " + str(index_exploit_noscan))


# Create exploit packets
print("\nCreating noscan-exploit packets...")

print("Opening Metasploit...")
openMetasploit()
print("Setting exploit parameters...")
setupExploit()

for index in range(index_exploit_noscan, index_exploit_noscan + packet_num):
	print("Packet no. " + str(index))
	packet_file = folder_exploit_noscan + "/exploit_noscan" + str(index)
	runTcpdump(packet_file)
	runExploit()
	time.sleep(delay_exploit)

exitMetasploit()

"""

# Write new index to index file
index_file_exploit_noscan.write(str(index_exploit + packet_num))

# Create safe packets
print("\nCreating safe packets...")

for index in range(index_safe, index_safe + packet_num):
	print("Packet no. " + str(index))
	packet_file = folder_safe + "/safe" + str(index)
	runTcpdump(packet_file)
	runRemmina()
	stopTcpdump()

# Write new index to index file
index_file_safe.write(str(index_safe + packet_num))

for index in range(index_0xeb, index_0xeb + packet_num):
	print("Packet no. " + str(index))
	packet_file = folder_0xeb + "/0xeb" + str(index)
	runTcpdump(packet_file)
	run0xeb()
	time.sleep(1)

for index in range(index_rdesktop, index_rdesktop + packet_num):
	time.sleep(1)
	print("Packet no. " + str(index))
	packet_file = folder_rdesktop + "/rdesktop" + str(index)
	runTcpdump(packet_file)
	runRDesktop()
	stopTcpdump()
"""

print("Done... :)")
