"""
Phase 2.5 -- Section 4: Final Test Evaluation (one-time, safety-guarded).

Uses: retrained models (Section 2) + thresholds_frozen_v2.json (Section 3).
Reads data/test.json exactly once.

Reports per category:
  - Metrics at BLOCK threshold
  - Metrics at REVIEW threshold
  - Indirect-Jailbreak metrics (new -- now has 5 test positives)
  - v1 vs v2 comparison on BLOCK-threshold precision

GUARD: If evaluation/results/final_test_metrics_v2.json exists, exit immediately.
"""

import os, sys, json, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RESULTS_DIR  = "evaluation/results"
GUARD_PATH   = os.path.join(RESULTS_DIR, "final_test_metrics_v2.json")
FROZEN_V2    = os.path.join(RESULTS_DIR, "thresholds_frozen_v2.json")
FROZEN_V1    = os.path.join(RESULTS_DIR, "thresholds_frozen.json")
CATEGORIES   = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]

# ── Safety guard ──────────────────────────────────────────────────────────────
if os.path.exists(GUARD_PATH):
    print("=" * 70)
    print("GUARD: final_test_metrics_v2.json already exists.")
    print("This evaluation was already run. To re-run, delete the guard file.")
    print("=" * 70)
    with open(GUARD_PATH, encoding="utf-8") as f:
        d = json.load(f)
    print("Existing results:")
    for cat, m in d["categories"].items():
        print(f"  {cat:<25}: F1@BLK={m['F1_at_block']:.3f}  "
              f"P@BLK={m['Precision_at_block']:.3f}  R@BLK={m['Recall_at_block']:.3f}")
    sys.exit(0)

print("=" * 70)
print("CipherGuard Phase 2.5 -- Final Test Evaluation (ONE-TIME RUN)")
print(f"Timestamp: {datetime.datetime.now().isoformat()}")
print("=" * 70)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, thr):
    yt = np.array(y_true)
    yp = np.array(y_prob)
    pred = (yp >= thr).astype(int)
    p = float(precision_score(yt, pred, zero_division=0))
    r = float(recall_score(yt, pred, zero_division=0))
    f = float(f1_score(yt, pred, zero_division=0))
    n_pos = int(yt.sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = None
    return {"Precision": round(p,3), "Recall": round(r,3), "F1": round(f,3),
            "ROC_AUC": round(auc,3) if auc else None,
            "TP": tp, "FP": fp, "FN": fn, "n_pos": n_pos, "threshold": thr}


# ── Load models and test data ──────────────────────────────────────────────
print("\nLoading models ...")
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from evaluation.build_dataset import get_category_labels

model_a = ModelFamilyA.load("models")
model_b = ModelFamilyB.load("models")

print("Loading test split (data/test.json) ...")
with open("data/test.json", encoding="utf-8") as f:
    test_data = json.load(f)

texts    = [d["text"] for d in test_data]
surfaces = [d.get("surface", "direct") for d in test_data]
n        = len(test_data)
print(f"  n={n} samples")

direct_idx   = [i for i, s in enumerate(surfaces) if s == "direct"]
indirect_idx = [i for i, s in enumerate(surfaces) if s == "indirect"]
print(f"  direct={len(direct_idx)}, indirect={len(indirect_idx)}")

# ── Compute Strategy D probabilities ──────────────────────────────────────
print("\nRunning inference (Strategy D -- max(LogReg, MLP)) ...")
batch_a = model_a.score_batch(texts, surfaces)
batch_b = model_b.score_batch(texts, surfaces)
probs   = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a, batch_b)]
           for cat in CATEGORIES}
y_true  = {cat: get_category_labels(test_data, cat) for cat in CATEGORIES}

# ── Load thresholds ─────────────────────────────────────────────────────────
frozen_v2 = load_json(FROZEN_V2)["thresholds"]
frozen_v1 = load_json(FROZEN_V1)["thresholds"] if os.path.exists(FROZEN_V1) else {}

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS AT BLOCK THRESHOLD (auto-reject, no human review)")
print("=" * 70)
print(f"\n{'Category':<25} {'V2_BLK':>7} {'P@BLK':>7} {'R@BLK':>7} {'F1@BLK':>7} "
      f"{'AUC':>7} {'TP':>4} {'FN':>4}  | V1 P@BLK -> V2 P@BLK")
print("-" * 100)

cat_results = {}
for cat in CATEGORIES:
    blk_v2 = frozen_v2[cat]["block"]
    m_blk  = compute_metrics(y_true[cat], probs[cat], blk_v2)

    # v1 precision at v1 block threshold (for comparison)
    v1_blk = frozen_v1.get(cat, {}).get("block", "N/A")
    if isinstance(v1_blk, (int, float)):
        m_v1_blk = compute_metrics(y_true[cat], probs[cat], v1_blk)
        v1_p_str = f"{m_v1_blk['Precision']:.3f}"
    else:
        v1_p_str = "N/A"

    print(f"  {cat:<23} {blk_v2:>7.2f} {m_blk['Precision']:>7.3f} {m_blk['Recall']:>7.3f} "
          f"{m_blk['F1']:>7.3f} {m_blk['ROC_AUC'] or 0:>7.3f} "
          f"{m_blk['TP']:>4} {m_blk['FN']:>4}  | {v1_p_str} -> {m_blk['Precision']:.3f}")

    rev_v2 = frozen_v2[cat]["review"]
    m_rev  = compute_metrics(y_true[cat], probs[cat], rev_v2)
    cat_results[cat] = {
        "block_threshold": blk_v2, "review_threshold": rev_v2,
        "F1_at_block":    m_blk["F1"],
        "Precision_at_block": m_blk["Precision"],
        "Recall_at_block": m_blk["Recall"],
        "ROC_AUC":        m_blk["ROC_AUC"],
        "TP": m_blk["TP"], "FP": m_blk["FP"], "FN": m_blk["FN"],
        "n_pos": m_blk["n_pos"],
        "F1_at_review":        m_rev["F1"],
        "Precision_at_review": m_rev["Precision"],
        "Recall_at_review":    m_rev["Recall"],
        "v1_Precision_at_block": float(v1_p_str) if v1_p_str != "N/A" else None,
    }

macro_f1_v2 = round(float(np.mean([cat_results[c]["F1_at_block"] for c in CATEGORIES])), 3)
print(f"\n  Macro-F1 @ BLOCK thresholds: {macro_f1_v2}")

print("\n" + "=" * 70)
print("RESULTS AT REVIEW THRESHOLD (flagged for human review)")
print("=" * 70)
print(f"\n{'Category':<25} {'V2_REV':>7} {'P@REV':>7} {'R@REV':>7} {'F1@REV':>7}")
print("-" * 60)
for cat in CATEGORIES:
    r = cat_results[cat]
    print(f"  {cat:<23} {r['review_threshold']:>7.2f} "
          f"{r['Precision_at_review']:>7.3f} {r['Recall_at_review']:>7.3f} "
          f"{r['F1_at_review']:>7.3f}")

# ── Direct/Indirect breakdown ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIRECT vs INDIRECT SURFACE BREAKDOWN (test, at BLOCK threshold)")
print("=" * 70)

surface_results = {}
for cat in ["Jailbreak", "Prompt Injection"]:
    blk = frozen_v2[cat]["block"]
    surface_results[cat] = {}
    for surf_name, idx in [("direct", direct_idx), ("indirect", indirect_idx)]:
        yt_s = [y_true[cat][i] for i in idx]
        yp_s = [probs[cat][i]  for i in idx]
        n_pos_s = sum(yt_s)
        if n_pos_s == 0:
            print(f"  {cat:<25} {surf_name:<10}: 0 positives -- skipping")
            surface_results[cat][surf_name] = {"n_pos": 0, "note": "no positives"}
            continue
        m_s = compute_metrics(yt_s, yp_s, blk)
        print(f"  {cat:<25} {surf_name:<10}: "
              f"P={m_s['Precision']:.3f} R={m_s['Recall']:.3f} F1={m_s['F1']:.3f} "
              f"  n={len(idx)} pos={n_pos_s}")
        surface_results[cat][surf_name] = m_s

# Explicit indirect-Jailbreak block (new in Phase 2.5)
print("\n--- Indirect-Jailbreak (key new metric) ---")
jb_indirect = surface_results.get("Jailbreak", {}).get("indirect", {})
if jb_indirect.get("n_pos", 0) > 0:
    print(f"  Indirect-Jailbreak @ BLOCK={frozen_v2['Jailbreak']['block']}:")
    print(f"    Precision={jb_indirect['Precision']:.3f}  "
          f"Recall={jb_indirect['Recall']:.3f}  "
          f"F1={jb_indirect['F1']:.3f}  "
          f"n_pos={jb_indirect['n_pos']}")
    print(f"  NOTE: n_pos={jb_indirect['n_pos']} is small -- treat F1 as point estimate.")
else:
    print("  Indirect-Jailbreak: 0 positives in test (unexpected -- check split)")

# ── Save results ──────────────────────────────────────────────────────────
output = {
    "timestamp": datetime.datetime.now().isoformat(),
    "strategy": "D_LR_MLP_MAX",
    "thresholds_version": "v2",
    "thresholds_file": FROZEN_V2,
    "n_test": n,
    "n_direct": len(direct_idx),
    "n_indirect": len(indirect_idx),
    "macro_F1_at_block": macro_f1_v2,
    "categories": cat_results,
    "surface_breakdown": surface_results,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(GUARD_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved (guard file) -> {GUARD_PATH}")

import csv
csv_path = os.path.join(RESULTS_DIR, "final_test_summary_v2.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Category","BLK_thr","P@BLK","R@BLK","F1@BLK","AUC","TP","FN",
                "REV_thr","P@REV","R@REV","F1@REV","V1_P@BLK"])
    for cat in CATEGORIES:
        r = cat_results[cat]
        w.writerow([cat, r["block_threshold"],
                    r["Precision_at_block"], r["Recall_at_block"], r["F1_at_block"],
                    r["ROC_AUC"], r["TP"], r["FN"],
                    r["review_threshold"],
                    r["Precision_at_review"], r["Recall_at_review"], r["F1_at_review"],
                    r.get("v1_Precision_at_block","N/A")])
print(f"CSV saved -> {csv_path}")

print("\n" + "=" * 70)
print(f"Phase 2.5 final test evaluation COMPLETE.")
print(f"Macro-F1 @ BLOCK (v2 thresholds): {macro_f1_v2}")
print("=" * 70)
