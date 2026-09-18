import json, os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
for split, path in [("train","data/train.json"),("val","data/val.json"),("test","data/test.json")]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    indirect_jb = [s for s in d if s.get("surface")=="indirect" and s["labels"]["Jailbreak"]==1]
    direct_jb   = [s for s in d if s.get("surface")=="direct"   and s["labels"]["Jailbreak"]==1]
    indirect_pi = [s for s in d if s.get("surface")=="indirect" and s["labels"]["Prompt Injection"]==1]
    indirect_n  = [s for s in d if s.get("surface")=="indirect"]
    print(f"{split} (n={len(d)}): indirect_JB_pos={len(indirect_jb)}, direct_JB_pos={len(direct_jb)}, indirect_PI_pos={len(indirect_pi)}, total_indirect={len(indirect_n)}")
