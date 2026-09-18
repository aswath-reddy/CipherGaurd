"""
Diagnostic: three-tier operating-point analysis + LogReg top-feature inspection.
Answers:
  1. Are the final test P/R/F1 at BLOCK or REVIEW threshold?
  2. What does precision look like at each operating point?
  3. Are LogReg high-weight features semantic or template artifacts?
"""
import json, os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from sklearn.metrics import precision_score, recall_score, f1_score
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from evaluation.build_dataset import get_category_labels

CATS = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]

# ── Load test data ────────────────────────────────────────────────────────────
with open("data/test.json", encoding="utf-8") as f:
    test_data = json.load(f)
texts    = [d["text"] for d in test_data]
surfaces = [d.get("surface", "direct") for d in test_data]

model_a = ModelFamilyA.load("models")
model_b = ModelFamilyB.load("models")
batch_a = model_a.score_batch(texts, surfaces)
batch_b = model_b.score_batch(texts, surfaces)

y_true = {cat: np.array(get_category_labels(test_data, cat)) for cat in CATS}
probs  = {cat: np.array([max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a, batch_b)])
          for cat in CATS}

with open("evaluation/results/thresholds_frozen.json") as f:
    frozen = json.load(f)["thresholds"]

# ── Three-tier analysis ───────────────────────────────────────────────────────
print("=" * 90)
print("THREE-TIER OPERATING-POINT ANALYSIS  (test n=140)")
print("BLOCK col  : score >= block_thr  --> auto-blocked, no human sees it")
print("FLAGGED col: score >= review_thr --> sent for any action (REVIEW + BLOCK)")
print("=" * 90)
print()
hdr = (f"{'Category':<25} {'BLK':>5} {'REV':>5} {'N_blk':>6} {'N_flag':>7}"
       f"  | {'P@BLK':>6} {'R@BLK':>6} {'F1@BLK':>7}"
       f"  | {'P@REV':>6} {'R@REV':>6} {'F1@REV':>7}")
print(hdr)
print("-" * 100)

results = {}
for cat in CATS:
    blk = frozen[cat]["block"]
    rev = frozen[cat]["review"]
    yt  = y_true[cat]
    yp  = probs[cat]

    pred_blk  = (yp >= blk).astype(int)
    pred_flag = (yp >= rev).astype(int)

    def m(yt, yp):
        return (round(float(precision_score(yt, yp, zero_division=0)), 3),
                round(float(recall_score(yt, yp, zero_division=0)), 3),
                round(float(f1_score(yt, yp, zero_division=0)), 3))

    pb, rb, fb = m(yt, pred_blk)
    pf, rf, ff = m(yt, pred_flag)
    n_blk  = int(pred_blk.sum())
    n_flag = int(pred_flag.sum())
    n_pos  = int(yt.sum())

    results[cat] = {"block": {"P": pb, "R": rb, "F1": fb, "n_blocked": n_blk},
                    "flagged": {"P": pf, "R": rf, "F1": ff, "n_flagged": n_flag},
                    "n_true_pos": n_pos}

    print(f"{cat:<25} {blk:>5.2f} {rev:>5.2f} {n_blk:>6} {n_flag:>7}"
          f"  | {pb:>6.3f} {rb:>6.3f} {fb:>7.3f}"
          f"  | {pf:>6.3f} {rf:>6.3f} {ff:>7.3f}"
          f"   n_pos={n_pos}")

print()
print("INTERPRETATION:")
for cat in CATS:
    r = results[cat]
    n_pos = r["n_true_pos"]
    n_blk = r["block"]["n_blocked"]
    pb    = r["block"]["P"]
    fp_blk = n_blk - round(pb * n_blk)
    print(f"  {cat:<25}: {n_blk} auto-blocked, {n_pos} real attacks "
          f"-> ~{fp_blk} legitimate requests auto-blocked (precision={pb:.3f})")

# ── LogReg top feature inspection ────────────────────────────────────────────
print()
print("=" * 90)
print("LOGREG TOP-10 FEATURES PER CATEGORY  (positive-class weights)")
print("Diagnosing: semantic signal vs template artifact?")
print("=" * 90)

for cat in CATS:
    pipe = model_a.pipelines.get(cat)
    if pipe is None:
        continue
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(pipe)
    except Exception:
        print(f"\n{cat}: UNFITTED"); continue

    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]

    vocab = {v: k for k, v in tfidf.vocabulary_.items()}
    coefs = clf.coef_[0]
    top10_pos = np.argsort(coefs)[-10:][::-1]
    top10_neg = np.argsort(coefs)[:5]

    print(f"\n[{cat}]  (train pos={int(y_true[cat].sum())} test positives)")
    print("  Top positive-weight features (model says ATTACK):")
    for idx in top10_pos:
        print(f"    {coefs[idx]:>+8.3f}  '{vocab.get(idx,'?')}'")
    print("  Top negative-weight features (model says BENIGN):")
    for idx in top10_neg:
        print(f"    {coefs[idx]:>+8.3f}  '{vocab.get(idx,'?')}'")
