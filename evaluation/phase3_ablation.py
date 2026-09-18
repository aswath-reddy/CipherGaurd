"""
Phase 3 - Section 17: Ablation Studies.

Covers:
  A. Dataset ablation: pilot only / public HF only / +PII+indirect / full
  B. Model ablation: A-only / B-only / DeBERTa / specialists / D (from cached results)
  C. Fusion ablation: MAX / weighted avg / stacking (from cached ablation_val_results.json)
  D. Policy ablation: shared v1 threshold vs v2 dual-objective
  E. Attribution ablation: beam=3/5/10, cap=4/6/8 on a small fixed set

For A-C: reuse existing results from Phase 2 ablation where possible.
For D: compare v1 and v2 threshold precision/recall from saved result files.
For E: run beam search on 5 fixed positives with varying beam/cap settings.
"""

import os, sys, json, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from src.attribution.beam_search import run_removal_beam_search
from evaluation.build_dataset import get_category_labels

CATEGORIES  = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR = "evaluation/results"
BLOCK_THR_V2= {"Jailbreak":0.45,"Prompt Injection":0.45,"PII Leakage":0.45,
               "Malicious Tools":0.45,"Hate/Toxicity":0.55}

print("=" * 65)
print("CipherGuard Phase 3 -- Section 17: Ablation Studies")
print("=" * 65)

ablation_output = {}

# ────────────────────────────────────────────────────────────────────────────────
# A. DATASET ABLATION
# Using val macro-F1 as proxy. We re-score val with Strategy D (already trained
# on the FULL dataset). For sub-dataset ablations we can only report what training
# data the Phase 2 models saw vs what they would see -- since we can't retrain
# all ablation variants here, we report the known composition + note what
# training on sub-datasets would be expected to affect.
# ────────────────────────────────────────────────────────────────────────────────
print("\n--- A. Dataset Ablation ---")
# Dataset composition summary from build_expanded_dataset.py output:
dataset_ablation = {
    "note": ("Full retrain on each subset would be needed for exact numbers. "
             "Numbers below are composition facts + expected impact, not measured val F1."),
    "configurations": {
        "pilot_only": {
            "n": 85, "sources": ["pilot"],
            "n_jailbreak_pos": 30, "n_pi_pos": 20,
            "note": "Phase 1 pilot dataset only. Extremely small; high variance expected."
        },
        "public_hf_only": {
            "n": 550, "sources": ["deepset/prompt-injections", "tweet_eval/hate"],
            "n_pi_pos": 350, "n_hate_pos": 100, "n_jailbreak_pos": 0,
            "note": "No Jailbreak or PII labels in public HF datasets. JB/PII classifiers would fail."
        },
        "public_hf_plus_pii_indirect": {
            "n": 761, "sources": ["deepset/prompt-injections", "tweet_eval/hate",
                                   "cipherguard_pii_synthetic", "cipherguard_indirect_synthetic",
                                   "cipherguard_indirect_jailbreak_synthetic"],
            "note": "Adds PII and indirect surface coverage. Still no malicious tools."
        },
        "full_dataset": {
            "n": 971, "sources": ["all"],
            "n_jailbreak": 140, "n_pi": 160, "n_pii": 68,
            "n_mt": 76, "n_hate": 100,
            "val_macro_f1_strategy_D": 0.788,
            "note": "Full Phase 2.5 dataset. This is the trained model."
        }
    }
}
print("  [COMPOSITION REPORTED -- retrain on sub-datasets NOT RUN (time budget)]")
for cfg, d in dataset_ablation["configurations"].items():
    val_f1_str = f"  val_macro_F1={d['val_macro_f1_strategy_D']:.3f}" if "val_macro_f1_strategy_D" in d else ""
    print(f"  {cfg:<40}: n={d['n']}{val_f1_str}")
ablation_output["dataset"] = dataset_ablation

# ────────────────────────────────────────────────────────────────────────────────
# B+C. MODEL AND FUSION ABLATION (from cached ablation_val_results.json)
# ────────────────────────────────────────────────────────────────────────────────
print("\n--- B+C. Model & Fusion Ablation (from cached ablation_val_results.json) ---")
abl_path = os.path.join(RESULTS_DIR, "ablation_val_results.json")
if os.path.exists(abl_path):
    abl_raw = json.load(open(abl_path, encoding="utf-8"))
    abl = abl_raw.get("strategies", abl_raw)   # unwrap if nested
    model_fusion_ablation = {}
    for strat, cats in abl.items():
        if not isinstance(cats, dict): continue
        f1s = [cats.get(c, {}).get("F1", 0) for c in CATEGORIES if isinstance(cats.get(c), dict)]
        macro = round(float(np.mean(f1s)), 4) if f1s else None
        model_fusion_ablation[strat] = {
            "macro_F1_val": macro,
            "per_category": {c: cats.get(c, {}).get("F1") for c in CATEGORIES
                             if isinstance(cats.get(c), dict)}
        }
        print(f"  {strat:<25}: macro-F1={macro:.4f}" if macro else f"  {strat}: N/A")
    ablation_output["model_and_fusion"] = model_fusion_ablation
else:
    print("  ablation_val_results.json NOT FOUND -- run phase2_ensemble_ablation.py first")
    ablation_output["model_and_fusion"] = {"error": "file not found"}

# ────────────────────────────────────────────────────────────────────────────────
# D. POLICY ABLATION: v1 (shared threshold, single-objective) vs v2 (dual-objective)
# ────────────────────────────────────────────────────────────────────────────────
print("\n--- D. Policy Ablation: v1 vs v2 thresholds ---")
v1_path = os.path.join(RESULTS_DIR, "thresholds_frozen.json")
v2_path = os.path.join(RESULTS_DIR, "thresholds_frozen_v2.json")

mA = ModelFamilyA.load("models")
mB = ModelFamilyB.load("models")
test_data  = json.load(open("data/test.json", encoding="utf-8"))
texts_t    = [d["text"] for d in test_data]
surfaces_t = [d.get("surface","direct") for d in test_data]
y_true_t   = {cat: get_category_labels(test_data, cat) for cat in CATEGORIES}
batch_a_t  = mA.score_batch(texts_t, surfaces_t)
batch_b_t  = mB.score_batch(texts_t, surfaces_t)
probs_t    = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a_t, batch_b_t)]
              for cat in CATEGORIES}

def eval_threshold(probs, y_true, thresholds_dict):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    results = {}
    for cat in CATEGORIES:
        thr = thresholds_dict.get(cat, {}).get("block", 0.5)
        yt = np.array(y_true[cat])
        yp = np.array(probs[cat])
        pred = (yp >= thr).astype(int)
        results[cat] = {
            "threshold": thr,
            "Precision": round(float(precision_score(yt,pred,zero_division=0)),3),
            "Recall":    round(float(recall_score(yt,pred,zero_division=0)),3),
            "F1":        round(float(f1_score(yt,pred,zero_division=0)),3),
        }
    macro = round(float(np.mean([results[c]["F1"] for c in CATEGORIES])),3)
    return results, macro

policy_ablation = {}
if os.path.exists(v1_path):
    v1 = json.load(open(v1_path, encoding="utf-8"))["thresholds"]
    res_v1, macro_v1 = eval_threshold(probs_t, y_true_t, v1)
    policy_ablation["v1_single_objective"] = {"macro_F1_test": macro_v1, "per_category": res_v1}
    print(f"  v1 (single-objective, recall-weighted): macro-F1={macro_v1:.3f}")
    for cat in CATEGORIES:
        r = res_v1[cat]
        print(f"    {cat:<25}: thr={r['threshold']:.2f}  P={r['Precision']:.3f}  R={r['Recall']:.3f}  F1={r['F1']:.3f}")
else:
    print("  v1 thresholds NOT FOUND")

if os.path.exists(v2_path):
    v2 = json.load(open(v2_path, encoding="utf-8"))["thresholds"]
    res_v2, macro_v2 = eval_threshold(probs_t, y_true_t, v2)
    policy_ablation["v2_dual_objective"] = {"macro_F1_test": macro_v2, "per_category": res_v2}
    print(f"\n  v2 (dual-objective, BLOCK precision-weighted): macro-F1={macro_v2:.3f}")
    for cat in CATEGORIES:
        r = res_v2[cat]
        print(f"    {cat:<25}: thr={r['threshold']:.2f}  P={r['Precision']:.3f}  R={r['Recall']:.3f}  F1={r['F1']:.3f}")

ablation_output["policy"] = policy_ablation

# ────────────────────────────────────────────────────────────────────────────────
# E. ATTRIBUTION ABLATION: beam x cap combinations on 5 fixed positives
# ────────────────────────────────────────────────────────────────────────────────
print("\n--- E. Attribution Ablation: beam_width x removal_cap ---")

val_data = json.load(open("data/val.json", encoding="utf-8"))
texts_v  = [d["text"] for d in val_data]
surfaces_v = [d.get("surface","direct") for d in val_data]
batch_a_v = mA.score_batch(texts_v, surfaces_v)
batch_b_v = mB.score_batch(texts_v, surfaces_v)
probs_v   = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a_v, batch_b_v)]
             for cat in CATEGORIES}

fixed_positives = []
for i, (text, surf) in enumerate(zip(texts_v, surfaces_v)):
    for cat in CATEGORIES:
        if probs_v[cat][i] >= BLOCK_THR_V2[cat]:
            fixed_positives.append({"text": text, "surface": surf, "category": cat, "score": probs_v[cat][i]})
            break
    if len(fixed_positives) >= 5:
        break

BEAM_WIDTHS  = [3, 5, 10]
REMOVAL_CAPS = [4, 6, 8]
attr_grid = {}

print(f"  Running {len(BEAM_WIDTHS)*len(REMOVAL_CAPS)} configs on {len(fixed_positives)} samples ...")
for bw in BEAM_WIDTHS:
    for cap in REMOVAL_CAPS:
        key = f"beam{bw}_cap{cap}"
        flip_rates = []
        latencies  = []
        n_removed_list = []
        for p in fixed_positives:
            cat = p["category"]
            thr = BLOCK_THR_V2[cat]
            def sfn(texts, cat=cat, surf=p["surface"]):
                aa = mA.score_batch(texts, [surf]*len(texts))
                bb = mB.score_batch(texts, [surf]*len(texts))
                return [max(a.get(cat,0),b.get(cat,0)) for a,b in zip(aa,bb)]
            res = run_removal_beam_search(p["text"], sfn, thr, beam_width=bw, removal_cap=cap)
            flip_rates.append(res.flipped)
            latencies.append(res.search_latency_ms)
            if res.flipped:
                n_removed_list.append(len(res.removed_tokens))
        attr_grid[key] = {
            "beam_width": bw, "removal_cap": cap,
            "flip_rate": round(float(np.mean(flip_rates)),2),
            "avg_latency_ms": round(float(np.mean(latencies)),1),
            "avg_tokens_removed": round(float(np.mean(n_removed_list)) if n_removed_list else 0,2),
        }

print(f"\n  {'Config':<18} {'Flip':>6} {'AvgLatMs':>10} {'AvgRemoved':>12}")
print(f"  {'-'*50}")
for key, d in attr_grid.items():
    print(f"  {key:<18} {d['flip_rate']:>6.2f} {d['avg_latency_ms']:>10.1f} {d['avg_tokens_removed']:>12.2f}")
ablation_output["attribution"] = attr_grid

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR,"ablation_phase3.json"),"w",encoding="utf-8") as f:
    json.dump({"timestamp":datetime.datetime.now().isoformat(), **ablation_output}, f, indent=2)
print("\nSaved -> evaluation/results/ablation_phase3.json")
print("\nSection 17 DONE")
