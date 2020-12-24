import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img
import numpy as np

exploit_dir = os.listdir("..\\Datatest\\exploit")
new_exploit_dir = os.listdir("..\\Datatest\\0xeb")
safe_dir = os.listdir("..\\Datatest\\safe")
safe_diff_dir = os.listdir("..\\Datatest\\safe_diff")

# for f in range(len(exploit_dir)):
#     sniff2img(os.path.join("..\\Datatest\\exploit", exploit_dir[f]),
#               os.path.join("..\\Datatest\\exploit_parsed_l100", str(f)), 100, "l", 100)
#     print("finished", f, "meta packets")
# packet_exploit = np.load("..\\Datatest\\exploit_parsed_l100\\0.npy")
# print(packet_exploit)
#
# for f in range(len(new_exploit_dir)):
#     sniff2img(os.path.join("..\\Datatest\\0xeb", new_exploit_dir[f]),
#               os.path.join("..\\Datatest\\new_exploit_parsed_l100", str(f)), 100, "l", 100)
#     print("finished", f, "0xeb packets")
# packet_new_exploit = np.load("..\\Datatest\\new_exploit_parsed_l100\\0.npy")
# print(packet_new_exploit)
#
# for f in range(len(safe_dir)):
#     sniff2img(os.path.join("..\\Datatest\\safe", safe_dir[f]),
#               os.path.join("..\\Datatest\\safe_parsed_l100", str(f + len(exploit_dir))), 100, "l", 100)
#     print("finished", f, "old packets")
# packet_safe = np.load("..\\Datatest\\safe_parsed_l100\\" + str(len(exploit_dir)) + ".npy")
# print(packet_safe)
#
# for f in range(len(safe_diff_dir)):
#     sniff2img(os.path.join("..\\Datatest\\safe_diff", safe_diff_dir[f]),
#               os.path.join("..\\Datatest\\safe_diff_parsed_l100", str(f + len(exploit_dir))), 100, "l", 100)
#     print("finished", f, "new packets")
packet_safe_diff = np.load("..\\Datatest\\safe_diff_parsed_l100\\" + str(len(exploit_dir)) + ".npy")
print(packet_safe_diff)

print("Starting dataset")

exploit_dir = os.listdir("..\\Dataset\\exploit")
safe_dir = os.listdir("..\\Dataset\\safe")
safe_diff_dir = os.listdir("..\\Dataset\\safe_diff")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("..\\Dataset\\exploit", exploit_dir[f]),
              os.path.join("..\\Dataset\\exploit_parsed_last_100", str(f)), 100, "l", 100)
    print("finished", f, "meta packets")
packet_exploit = np.load("..\\Dataset\\exploit_parsed_last_100\\0.npy")
print(packet_exploit)

for f in range(len(safe_dir)):
    sniff2img(os.path.join("..\\Dataset\\safe", safe_dir[f]),
              os.path.join("..\\Dataset\\safe_parsed_last_100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "old packets")
packet_safe = np.load("..\\Dataset\\safe_parsed_last_100\\" + str(len(exploit_dir)) + ".npy")
print(packet_safe)

for f in range(len(safe_diff_dir)):
    sniff2img(os.path.join("..\\Dataset\\safe_diff", safe_diff_dir[f]),
              os.path.join("..\\Dataset\\safe_diff_parsed_last_100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "new packets")
packet_safe_diff = np.load("..\\Dataset\\safe_diff_parsed_last_100\\" + str(len(exploit_dir)) + ".npy")
print(packet_safe_diff)