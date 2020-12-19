import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img

exploit_dir = os.listdir("../Datatest/metasploit")
xeb_dir = os.listdir("../Datatest/0xeb")
safe_old_dir = os.listdir("../Datatest/Remmina")
safe_new_dir = os.listdir("../Datatest/RDesktop")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("../Datatest/metasploit", exploit_dir[f]),
              os.path.join("../Datatest/metasploit_last100", str(f)), 100, "l", 100)
    print("finished", f, "meta packets")
for f in range(len(xeb_dir)):
    sniff2img(os.path.join("../Datatest/0xeb", xeb_dir[f]),
              os.path.join("../Datatest/0xeb_last100", str(f)), 100, "l", 100)
    print("finished", f, "0xeb packets")
for f in range(len(safe_old_dir)):
    sniff2img(os.path.join("../Datatest/Remmina", safe_old_dir[f]),
              os.path.join("../Datatest/Remmina_last100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "old packets")
for f in range(len(safe_new_dir)):
    sniff2img(os.path.join("../Datatest/RDesktop", safe_new_dir[f]),
              os.path.join("../Datatest/RDesktop_last100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "new packets")

exploit_dir = os.listdir("../Datatrain/Metasploit")
safe_old_dir = os.listdir("../Datatrain/Remmina")
safe_new_dir = os.listdir("../Datatrain/RDesktop")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("../Datatrain/Metasploit", exploit_dir[f]),
              os.path.join("../Datatrain/Metasploit_last100", str(f)), 100, "l", 100)
    print("finished", f, "metasploit packets")
for f in range(len(safe_old_dir)):
    sniff2img(os.path.join("../Datatrain/Remmina", safe_old_dir[f]),
              os.path.join("../Datatrain/Remmina_last100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "remmina packets")
for f in range(len(safe_new_dir)):
    sniff2img(os.path.join("../Datatrain/RDesktop", safe_new_dir[f]),
              os.path.join("../Datatrain/RDesktop_last100", str(f + len(exploit_dir))), 100, "l", 100)
    print("finished", f, "rdesktop packets")
