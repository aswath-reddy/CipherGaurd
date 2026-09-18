"""
Phase 3 - Section 18: Confusion Matrices and Error Analysis.

Runs Strategy D on the test split, saves per-category confusion matrices
and representative FP/FN examples (with input, GT label, predicted label,
score, surface, attack_type, router decision).
"""

import os, sys, json, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from evaluation.build_dataset import get_category_labels

CATEGORIES  = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR = "evaluation/results"
BLOCK_THR   = {"Jailbreak":0.45,"Prompt Injection":0.45,"PII Leakage":0.45,
               "Malicious Tools":0.45,"Hate/Toxicity":0.55}
REVIEW_THR  = {"Jailbreak":0.35,"Prompt Injection":0.40,"PII Leakage":0.40,
               "Malicious Tools":0.30,"Hate/Toxicity":0.40}
MAX_EXAMPLES = 5  # representative examples per FP/FN per category

print("=" * 65)
print("CipherGuard Phase 3 -- Section 18: Confusion Matrices")
print("=" * 65)

# Load models
mA = ModelFamilyA.load("models")
mB = ModelFamilyB.load("models")

# Load test split
test_data = json.load(open("data/test.json", encoding="utf-8"))
texts    = [d["text"] for d in test_data]
surfaces = [d.get("surface", "direct") for d in test_data]
atypes   = [d.get("attack_type", "unknown") for d in test_data]
n = len(test_data)

# Score
batch_a = mA.score_batch(texts, surfaces)
batch_b = mB.score_batch(texts, surfaces)
probs   = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a, batch_b)]
           for cat in CATEGORIES}
y_true  = {cat: get_category_labels(test_data, cat) for cat in CATEGORIES}

def router_decision(scores_vec):
    for cat in CATEGORIES:
        if scores_vec[cat] >= BLOCK_THR[cat]:
            return "BLOCK"
    for cat in CATEGORIES:
        if scores_vec[cat] >= REVIEW_THR[cat]:
            return "REVIEW"
    return "ALLOW"

# Build per-sample score dicts for router decision
sample_scores = [{cat: probs[cat][i] for cat in CATEGORIES} for i in range(n)]
decisions = [router_decision(s) for s in sample_scores]

print(f"\nTest split: n={n}")
print(f"\n{'Category':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 70)

matrices  = {}
error_ex  = {}

for cat in CATEGORIES:
    yt = np.array(y_true[cat])
    yp = np.array(probs[cat])
    pred = (yp >= BLOCK_THR[cat]).astype(int)

    tp = int(((pred==1)&(yt==1)).sum())
    fp = int(((pred==1)&(yt==0)).sum())
    fn = int(((pred==0)&(yt==1)).sum())
    tn = int(((pred==0)&(yt==0)).sum())
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1   = 2*prec*rec/max(1e-9,prec+rec)

    print(f"  {cat:<23} {tp:>4} {fp:>4} {fn:>4} {tn:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

    matrices[cat] = {"TP":tp,"FP":fp,"FN":fn,"TN":tn,
                     "Precision":round(prec,4),"Recall":round(rec,4),"F1":round(f1,4)}

    # Collect FP examples
    fp_idxs = [i for i in range(n) if pred[i]==1 and yt[i]==0]
    fn_idxs = [i for i in range(n) if pred[i]==0 and yt[i]==1]

    def make_example(i, kind):
        return {
            "kind": kind,
            "text_preview": texts[i][:200],
            "ground_truth": int(yt[i]),
            "predicted": int(pred[i]),
            "score": round(float(yp[i]),4),
            "surface": surfaces[i],
            "attack_type": atypes[i],
            "router_decision": decisions[i],
        }

    error_ex[cat] = {
        "FP": [make_example(i,"FP") for i in sorted(fp_idxs, key=lambda x: -probs[cat][x])[:MAX_EXAMPLES]],
        "FN": [make_example(i,"FN") for i in sorted(fn_idxs, key=lambda x: probs[cat][x])[:MAX_EXAMPLES]],
    }

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR,"confusion_matrices.json"),"w",encoding="utf-8") as f:
    json.dump({"timestamp":datetime.datetime.now().isoformat(),
               "n_test":n,"thresholds":BLOCK_THR,"matrices":matrices}, f, indent=2)
with open(os.path.join(RESULTS_DIR,"error_analysis.json"),"w",encoding="utf-8") as f:
    json.dump({"timestamp":datetime.datetime.now().isoformat(),
               "max_examples_per_type":MAX_EXAMPLES,"examples":error_ex}, f, indent=2)

# ── Print representative errors ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("REPRESENTATIVE FALSE POSITIVES (highest-scored FPs per category)")
print("=" * 65)
for cat in CATEGORIES:
    fps = error_ex[cat]["FP"]
    if not fps: print(f"  {cat}: no FPs"); continue
    ex = fps[0]
    print(f"\n  {cat} (score={ex['score']}, surface={ex['surface']}, "
          f"attack_type={ex['attack_type']}, decision={ex['router_decision']})")
    print(f"  TEXT: {ex['text_preview'][:150]}")

print("\n" + "=" * 65)
print("REPRESENTATIVE FALSE NEGATIVES (lowest-scored FNs per category)")
print("=" * 65)
for cat in CATEGORIES:
    fns = error_ex[cat]["FN"]
    if not fns: print(f"  {cat}: no FNs"); continue
    ex = fns[0]
    print(f"\n  {cat} (score={ex['score']}, surface={ex['surface']}, "
          f"attack_type={ex['attack_type']}, decision={ex['router_decision']})")
    print(f"  TEXT: {ex['text_preview'][:150]}")

print("\nSaved -> evaluation/results/confusion_matrices.json")
print("Saved -> evaluation/results/error_analysis.json")
print("\nSection 18 DONE")
