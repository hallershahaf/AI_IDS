import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img

exploit_dir = os.listdir("../Dataset/exploit")
safe_dir = os.listdir("../Dataset/safe")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("../Dataset/exploit", exploit_dir[f]), os.path.join("../Dataset/exploit_parsed_big", str(f)), 132, "n", 0)
    print("finished", f)
for f in range(len(safe_dir)):
    sniff2img(os.path.join("../Dataset/safe", safe_dir[f]), os.path.join("../Dataset/safe_parsed_big", str(f + len(exploit_dir))), 132, "o", 0)
    print("finished", f)
    
####################################################################################################################################
import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img

exploit_dir = os.listdir("../Datatest/exploit")
safe_dir = os.listdir("../Datatest/safe")

for f in range(len(safe_dir)):
    sniff2img(os.path.join("../Datatest/safe", safe_dir[f]), os.path.join("../Datatest/safe_moved_l_1", str(f + len(exploit_dir))), 128, "l", 1)
    print("finished",f)
for f in range(len(safe_dir)):
    sniff2img(os.path.join("../Datatest/safe", safe_dir[f]), os.path.join("../Datatest/safe_moved_l_2", str(f + len(exploit_dir))), 128, "l", 2)
    print("finished",f)
for f in range(len(safe_dir)):
    sniff2img(os.path.join("../Datatest/safe", safe_dir[f]), os.path.join("../Datatest/safe_moved_l_3", str(f + len(exploit_dir))), 128, "l", 3)
    print("finished",f)
    
####################################################################################################################################
import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img

exploit_dir = os.listdir("../Datatest/exploit")
safe_dir = os.listdir("../Datatest/safe_diff")

for f in range(len(safe_dir)):
    sniff2img(os.path.join("../Datatest/safe_diff", safe_dir[f]), os.path.join("../Datatest/safe_diff_parsed", str(f + len(exploit_dir))), 128, "l", 0)
    print("finished",f)

