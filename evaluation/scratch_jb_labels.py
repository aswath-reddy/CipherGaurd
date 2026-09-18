import sys
sys.path.insert(0, ".")
from evaluation.indirect_surface_dataset import INDIRECT_JAILBREAK_MALICIOUS

picks = [0, 3, 5, 12, 15, 25]
for i in picks:
    s = INDIRECT_JAILBREAK_MALICIOUS[i]
    print("[%d] %s" % (i, s["attack_type"]))
    print("     TEXT: %s" % s["text"][:220])
    print()
