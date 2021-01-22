import os
from AI_IDS.parsing_scripts.sniff2img_hast_i import sniff2img
from AI_IDS.parsing_scripts.delete_plain_text import delete_plain_text
from datetime import datetime as time

# This whole file is a draft we use to re-parse packets for various purposes
# It isn't actually needed to run the NN

"""
exploit_dir = os.listdir("../Datatest/Metasploit")
xeb_dir = os.listdir("../Datatest/0xeb")
safe_old_dir = os.listdir("../Datatest/Remmina")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("../Datatest/Metasploit", exploit_dir[f]),
              os.path.join("../Datatest/Metasploit_first200", str(f)), 200, "l", 0)
    print("finished", f, "meta packets")
for f in range(len(xeb_dir)):
    sniff2img(os.path.join("../Datatest/0xeb", xeb_dir[f]),
              os.path.join("../Datatest/0xeb_first200", str(f)), 200, "l", 0)
    print("finished", f, "0xeb packets")
for f in range(len(safe_old_dir)):
    sniff2img(os.path.join("../Datatest/Remmina", safe_old_dir[f]),
              os.path.join("../Datatest/Remmina_first200", str(f + len(exploit_dir))), 200, "l", 0)
    print("finished", f, "old packets")
for f in range(len(safe_new_dir)):
    sniff2img(os.path.join("../Datatest/RDesktop", safe_new_dir[f]),
              os.path.join("../Datatest/RDesktop_first200", str(f + len(exploit_dir))), 200, "l", 0)
    print("finished", f, "new packets")

exploit_dir = os.listdir("../Datatrain/Metasploit")
safe_old_dir = os.listdir("../Datatrain/Remmina")
safe_new_dir = os.listdir("../Datatrain/RDesktop")

for f in range(len(exploit_dir)):
    sniff2img(os.path.join("../Datatrain/Metasploit", exploit_dir[f]),
              os.path.join("../Datatrain/Metasploit_first200", str(f)), 200, "l", 0)
    print("finished", f, "meta packets")
for f in range(len(safe_old_dir)):
    sniff2img(os.path.join("../Datatrain/Remmina", safe_old_dir[f]),
              os.path.join("../Datatrain/Remmina_first200", str(f + len(exploit_dir))), 200, "l", 0)
    print("finished", f, "remmina packets")
for f in range(len(safe_new_dir)):
    sniff2img(os.path.join("../Datatrain/RDesktop", safe_new_dir[f]),
              os.path.join("../Datatrain/Remmina_first200", str(f + len(exploit_dir))), 200, "l", 0)
    print("finished", f, "rdesktop packets")
    """
"""
delete_plain_text(os.getcwd() + "\\..\\Datatrain\\Metasploit_first200", os.getcwd() + "\\..\\Datatrain\\Metasploit_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatrain\\Remmina_first200",    os.getcwd() + "\\..\\Datatrain\\Remmina_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatrain\\RDesktop_first200",   os.getcwd() + "\\..\\Datatrain\\RDesktop_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatest\\Metasploit_first200",  os.getcwd() + "\\..\\Datatest\\Metasploit_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatest\\0xeb_first200",        os.getcwd() + "\\..\\Datatest\\0xeb_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatest\\Remmina_first200",     os.getcwd() + "\\..\\Datatest\\Remmina_encrypted200", 200)
delete_plain_text(os.getcwd() + "\\..\\Datatest\\RDesktop_first200",    os.getcwd() + "\\..\\Datatest\\RDesktop_encrypted200", 200)
"""

