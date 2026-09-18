"""
Phase 2 -- Final Test Evaluation (ONE-TIME ONLY).

Loads frozen thresholds + best ensemble strategy, scores data/test.json,
and writes the final unbiased held-out test metrics.

SAFETY GUARD: Aborts if thresholds_frozen.json does not exist.
This script must never be run before thresholds are frozen.
It must never be run more than once for reporting purposes.

Outputs:
  evaluation/results/final_test_metrics.json
  evaluation/results/final_test_summary.csv
"""

import os, sys, json, csv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from evaluation.build_dataset import get_category_labels
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from src.classification.finetuned_deberta_classifier import FinetunedDebertaClassifier
from src.classification.toxic_bert_classifier import ToxicBertClassifier
from src.classification.presidio_classifier import PresidioClassifier

CATEGORIES   = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR  = "evaluation/results"
MODELS_DIR   = "models"
TEST_PATH    = "data/test.json"
FROZEN_PATH  = os.path.join(RESULTS_DIR, "thresholds_frozen.json")
BEST_PATH    = os.path.join(RESULTS_DIR, "best_strategy.json")

SPECIALIST_MAP = {
    "Jailbreak":       "deberta",
    "Prompt Injection":"deberta",
    "PII Leakage":     "presidio",
    "Malicious Tools": None,
    "Hate/Toxicity":   "toxicbert",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_probs(raw_scores, strategy_name):
    """Reconstruct ensemble probability scores for test set."""
    model_keys = ["model_a", "model_b", "deberta", "toxicbert", "presidio"]
    if strategy_name.startswith("A_"):
        return {cat: raw_scores["model_a"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("B_"):
        return {cat: raw_scores["model_b"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("C_"):
        probs = {}
        for cat in CATEGORIES:
            spec = SPECIALIST_MAP[cat]
            probs[cat] = raw_scores[spec][cat] if spec else [0.0]*len(raw_scores["model_a"][cat])
        return probs
    elif strategy_name.startswith("D_"):
        return {cat: [max(a,b) for a,b in zip(raw_scores["model_a"][cat], raw_scores["model_b"][cat])]
                for cat in CATEGORIES}
    elif strategy_name.startswith("G_"):
        import joblib
        clf_path = os.path.join(RESULTS_DIR, "meta_classifier_G.pkl")
        meta_clfs = joblib.load(clf_path)
        probs = {}
        for cat in CATEGORIES:
            if cat in meta_clfs:
                X = np.column_stack([np.array(raw_scores[mk][cat]) for mk in model_keys])
                probs[cat] = list(meta_clfs[cat].predict_proba(X)[:, 1])
            else:
                spec = SPECIALIST_MAP[cat]
                if spec:
                    probs[cat] = [max(a,b,s) for a,b,s in zip(
                        raw_scores["model_a"][cat], raw_scores["model_b"][cat], raw_scores[spec][cat])]
                else:
                    probs[cat] = [max(a,b) for a,b in zip(
                        raw_scores["model_a"][cat], raw_scores["model_b"][cat])]
        return probs
    elif strategy_name.startswith("F_"):
        weights = load_json(os.path.join(RESULTS_DIR, "fusion_weights_F.json"))
        probs = {}
        for cat in CATEGORIES:
            w   = weights.get(cat, {"w_a": 0.333, "w_b": 0.333, "w_spec": 0.334})
            spec = SPECIALIST_MAP[cat]
            a   = np.array(raw_scores["model_a"][cat])
            b   = np.array(raw_scores["model_b"][cat])
            s   = np.array(raw_scores[spec][cat]) if spec else np.zeros(len(a))
            probs[cat] = list(w["w_a"]*a + w["w_b"]*b + w["w_spec"]*s)
        return probs
    else:  # E or fallback
        probs = {}
        for cat in CATEGORIES:
            spec = SPECIALIST_MAP[cat]
            if spec:
                probs[cat] = [max(a,b,s) for a,b,s in zip(
                    raw_scores["model_a"][cat], raw_scores["model_b"][cat], raw_scores[spec][cat])]
            else:
                probs[cat] = [max(a,b) for a,b in zip(
                    raw_scores["model_a"][cat], raw_scores["model_b"][cat])]
        return probs


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    n_pos  = int(y_true.sum())
    n_neg  = int(len(y_true) - n_pos)
    if n_pos == 0:
        return {"note": "no positives", "n_pos": 0, "n_neg": n_neg}
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except Exception:
        tp = int(((y_pred==1)&(y_true==1)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())
    try:
        roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else None
    except Exception:
        roc = None
    try:
        pra = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else None
    except Exception:
        pra = None
    return {
        "n_pos":     n_pos, "n_neg": n_neg,
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "ROC_AUC":   round(float(roc), 4) if roc is not None else None,
        "PR_AUC":    round(float(pra), 4) if pra is not None else None,
    }


def main():
    # ------------------------------------------------------------------
    # Safety guard
    # ------------------------------------------------------------------
    if not os.path.exists(FROZEN_PATH):
        print("ABORT: thresholds_frozen.json not found.")
        print("Run phase2_threshold_search.py first to freeze thresholds.")
        sys.exit(1)

    frozen     = load_json(FROZEN_PATH)
    best       = load_json(BEST_PATH) if os.path.exists(BEST_PATH) else {}
    strategy   = best.get("best_strategy", frozen.get("strategy", "E_LR_MLP_Spec_MAX"))
    thresholds = frozen["thresholds"]

    print("=" * 70)
    print("CipherGuard Phase 2 -- FINAL TEST EVALUATION (one-time)")
    print(f"Strategy : {strategy}")
    print(f"Thresholds: frozen from val split")
    print("=" * 70)

    # Load test split
    test_data = load_json(TEST_PATH)
    texts     = [d["text"] for d in test_data]
    surfaces  = [d.get("surface", "direct") for d in test_data]
    y_true    = {cat: get_category_labels(test_data, cat) for cat in CATEGORIES}
    print(f"\nTest split: {len(test_data)} samples  "
          f"(direct={surfaces.count('direct')}, indirect={surfaces.count('indirect')})")

    # Load models
    print("\nLoading models ...")
    model_a   = ModelFamilyA.load(MODELS_DIR)
    model_b   = ModelFamilyB.load(MODELS_DIR)
    deberta   = FinetunedDebertaClassifier()
    toxicbert = ToxicBertClassifier()
    presidio  = PresidioClassifier()

    # Score test set
    print("Scoring test set ...")
    def sbatch(model, name):
        try:
            batch = model.score_batch(texts, surfaces)
        except Exception as e:
            print(f"  [WARNING] {name} score_batch failed: {e}; falling back.")
            batch = [model.score(t, s) for t, s in zip(texts, surfaces)]
        return {cat: [r.get(cat, 0.0) for r in batch] for cat in CATEGORIES}

    raw_test = {
        "model_a":   sbatch(model_a,   "Family A"),
        "model_b":   sbatch(model_b,   "Family B"),
        "deberta":   sbatch(deberta,   "DeBERTa"),
        "toxicbert": sbatch(toxicbert, "ToxicBERT"),
        "presidio":  sbatch(presidio,  "Presidio"),
    }
    probs = get_probs(raw_test, strategy)

    # ------------------------------------------------------------------
    # Compute metrics using frozen thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST METRICS (frozen thresholds)")
    print(f"{'Category':<25} {'Thr':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'ROC':>6} {'TP':>4} {'FN':>4}")
    print("-" * 75)

    results     = {"data_split": "test", "strategy": strategy,
                   "frozen_threshold_source": "val", "categories": {}}
    csv_rows    = []
    f1_scores   = []

    for cat in CATEGORIES:
        thr = thresholds[cat]["block"]
        m   = compute_metrics(y_true[cat], probs[cat], thr)
        results["categories"][cat] = m
        if "note" in m:
            print(f"  {cat:<23}: {m['note']}")
        else:
            roc = f"{m['ROC_AUC']:.3f}" if m["ROC_AUC"] else "N/A"
            print(f"  {cat:<23} {thr:>5.2f} {m['Precision']:>6.3f} {m['Recall']:>6.3f} "
                  f"{m['F1']:>6.3f} {m['Accuracy']:>6.3f} {roc:>6} {m['TP']:>4} {m['FN']:>4}")
            f1_scores.append(m["F1"])
            csv_rows.append({"category": cat, "split": "test", "strategy": strategy,
                             "threshold": thr, **{k: m[k] for k in
                             ["Precision","Recall","F1","Accuracy","ROC_AUC","PR_AUC","TP","FP","TN","FN"]}})

    macro_f1 = round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4)
    results["macro_F1_test"] = macro_f1
    print(f"\n  Macro-F1 (test): {macro_f1:.4f}")

    # Direct vs Indirect breakdown
    surfaces_arr = np.array(surfaces)
    direct_idx   = np.where(surfaces_arr == "direct")[0]
    indirect_idx = np.where(surfaces_arr == "indirect")[0]
    print(f"\n--- Direct vs Indirect (test: direct={len(direct_idx)}, indirect={len(indirect_idx)}) ---")
    surface_results = {}
    for cat in ["Jailbreak", "Prompt Injection"]:
        surface_results[cat] = {}
        thr = thresholds[cat]["block"]
        for surf_name, idx in [("direct", direct_idx), ("indirect", indirect_idx)]:
            if len(idx) == 0:
                continue
            yt_s = [y_true[cat][i] for i in idx]
            yp_s = [probs[cat][i]  for i in idx]
            # Per-surface threshold if available
            ps   = thresholds[cat].get("per_surface", {}).get(surf_name, {})
            thr_s = ps.get("block", thr)
            m_s  = compute_metrics(yt_s, yp_s, thr_s)
            surface_results[cat][surf_name] = m_s
            if "note" not in m_s:
                print(f"  {cat:<25} {surf_name:<10}: "
                      f"P={m_s['Precision']:.3f} R={m_s['Recall']:.3f} "
                      f"F1={m_s['F1']:.3f}  n={len(idx)} pos={m_s['n_pos']}")
    results["surface_breakdown"] = surface_results

    # Save
    out_path = os.path.join(RESULTS_DIR, "final_test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal test metrics saved -> {out_path}")

    csv_path = os.path.join(RESULTS_DIR, "final_test_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Final test CSV saved -> {csv_path}")

    print("\n" + "=" * 70)
    print("Phase 2 COMPLETE. These are the final unbiased test results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
