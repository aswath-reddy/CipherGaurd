"""
Phase 2 -- Section 8 & 9: Benchmark all individual models on the VALIDATION split.

Models evaluated:
  A. TF-IDF + Logistic Regression (Family A)
  B. TF-IDF + MLP (Family B)
  C. ProtectAI DeBERTa (zero-shot, Jailbreak/PI only)
  D. ToxicBERT (Hate/Toxicity only)
  E. Presidio (PII Leakage only)
  F. Current ensemble (MAX fusion)

Metrics per model per applicable category:
  Precision, Recall, F1, Accuracy, TP, FP, TN, FN, ROC-AUC, PR-AUC

Section 9: direct vs indirect breakdown for Jailbreak and Prompt Injection.

Outputs:
  evaluation/results/val_raw_scores.json       -- raw probability scores (for ablation)
  evaluation/results/benchmark_val_metrics.json
  evaluation/results/benchmark_summary.csv

GROUND RULE: data/test.json is never loaded here.
"""

import os
import sys
import json
import csv

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
from src.classification.risk_aggregator import RiskAggregator

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
MODELS_DIR  = "models"
RESULTS_DIR = "evaluation/results"
VAL_PATH    = "data/val.json"

# Which categories each specialist covers meaningfully
SPECIALIST_COVERAGE = {
    "deberta":    {"Jailbreak", "Prompt Injection"},
    "toxicbert":  {"Hate/Toxicity"},
    "presidio":   {"PII Leakage"},
}
SURFACE_BREAKDOWN_CATS = ["Jailbreak", "Prompt Injection"]


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Return full metric dict given true labels and probability scores."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0:
        return {"note": "no positives in split — all metrics undefined", "n_pos": 0, "n_neg": n_neg}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
        if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 \
        else _safe_cm(y_true, y_pred)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    acc   = accuracy_score(y_true, y_pred)

    # AUC — requires score variance
    try:
        roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    except Exception:
        pr_auc = None

    return {
        "n_pos":     n_pos,
        "n_neg":     n_neg,
        "TP":        int(tp),
        "FP":        int(fp),
        "TN":        int(tn),
        "FN":        int(fn),
        "Precision": round(float(prec), 4),
        "Recall":    round(float(rec), 4),
        "F1":        round(float(f1), 4),
        "Accuracy":  round(float(acc), 4),
        "ROC_AUC":   round(float(roc_auc), 4) if roc_auc is not None else None,
        "PR_AUC":    round(float(pr_auc), 4) if pr_auc is not None else None,
    }


def _safe_cm(y_true, y_pred):
    """Confusion matrix when only one class present in predictions."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tn, fp, fn, tp


def score_all(model, texts, surfaces, model_name):
    """Score all texts in batch; return dict of category -> list of probs."""
    print(f"  Scoring with {model_name} ({len(texts)} samples) ...")
    try:
        batch = model.score_batch(texts, surfaces)
    except Exception as e:
        print(f"  [WARNING] score_batch failed ({e}); falling back to individual scoring.")
        batch = [model.score(t, s) for t, s in zip(texts, surfaces)]
    # Transpose: list of dicts -> dict of lists
    return {cat: [r.get(cat, 0.0) for r in batch] for cat in CATEGORIES}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("CipherGuard Phase 2 -- Individual Model Benchmark (val split)")
    print("=" * 70)

    # -- Load val split --
    val_data = load_split(VAL_PATH)
    texts    = [d["text"] for d in val_data]
    surfaces = [d.get("surface", "direct") for d in val_data]
    print(f"\nValidation split loaded: {len(val_data)} samples")
    print(f"  direct={surfaces.count('direct')}, indirect={surfaces.count('indirect')}")

    # Ground-truth labels
    y_true = {cat: np.array(get_category_labels(val_data, cat)) for cat in CATEGORIES}

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("\nLoading models ...")
    print("  Loading Family A (LogReg) ...")
    model_a = ModelFamilyA.load(MODELS_DIR)

    print("  Loading Family B (MLP) ...")
    model_b = ModelFamilyB.load(MODELS_DIR)

    print("  Loading ProtectAI DeBERTa (via FinetunedDebertaClassifier) ...")
    deberta = FinetunedDebertaClassifier()

    print("  Loading ToxicBERT ...")
    toxicbert = ToxicBertClassifier()

    print("  Loading Presidio ...")
    presidio = PresidioClassifier()

    print("  Building ensemble (MAX fusion) ...")
    ensemble = RiskAggregator(
        model_a=model_a, model_b=model_b,
        deberta=deberta, presidio=presidio, toxic_bert=toxicbert,
        models_dir=MODELS_DIR
    )

    # -------------------------------------------------------------------------
    # Score all models
    # -------------------------------------------------------------------------
    print("\nRunning inference on validation set ...")
    scores_a        = score_all(model_a,   texts, surfaces, "Family A (LogReg)")
    scores_b        = score_all(model_b,   texts, surfaces, "Family B (MLP)")
    scores_deberta  = score_all(deberta,   texts, surfaces, "DeBERTa (ProtectAI)")
    scores_toxicbert= score_all(toxicbert, texts, surfaces, "ToxicBERT")
    scores_presidio = score_all(presidio,  texts, surfaces, "Presidio")

    print("  Scoring with MAX Ensemble ...")
    ens_batch = ensemble.aggregate_batch(texts, surfaces)
    scores_ensemble = {cat: [r["fused_scores"].get(cat, 0.0) for r in ens_batch] for cat in CATEGORIES}

    # Save raw scores for ablation use
    raw_scores = {
        "data_split": "val",
        "n": len(val_data),
        "model_a":    {cat: [round(v,4) for v in scores_a[cat]]         for cat in CATEGORIES},
        "model_b":    {cat: [round(v,4) for v in scores_b[cat]]         for cat in CATEGORIES},
        "deberta":    {cat: [round(v,4) for v in scores_deberta[cat]]   for cat in CATEGORIES},
        "toxicbert":  {cat: [round(v,4) for v in scores_toxicbert[cat]] for cat in CATEGORIES},
        "presidio":   {cat: [round(v,4) for v in scores_presidio[cat]]  for cat in CATEGORIES},
        "ensemble":   {cat: [round(v,4) for v in scores_ensemble[cat]]  for cat in CATEGORIES},
        "y_true":     {cat: y_true[cat].tolist()                         for cat in CATEGORIES},
        "surfaces":   surfaces,
    }
    raw_path = os.path.join(RESULTS_DIR, "val_raw_scores.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_scores, f, indent=2)
    print(f"\nRaw scores saved -> {raw_path}")

    # -------------------------------------------------------------------------
    # Compute metrics per model per category
    # -------------------------------------------------------------------------
    all_models = {
        "A_LogReg":  scores_a,
        "B_MLP":     scores_b,
        "C_DeBERTa": scores_deberta,
        "D_ToxicBERT": scores_toxicbert,
        "E_Presidio":  scores_presidio,
        "F_Ensemble":  scores_ensemble,
    }
    # Categories each model is meaningful for (others reported as N/A)
    model_coverage = {
        "A_LogReg":    set(CATEGORIES),
        "B_MLP":       set(CATEGORIES),
        "C_DeBERTa":   {"Jailbreak", "Prompt Injection"},
        "D_ToxicBERT": {"Hate/Toxicity"},
        "E_Presidio":  {"PII Leakage"},
        "F_Ensemble":  set(CATEGORIES),
    }

    results = {"data_split": "val", "threshold": 0.5, "models": {}}

    print("\n" + "=" * 70)
    print("SECTION 8 -- Per-Model Metrics (validation, threshold=0.5)")
    print("=" * 70)

    csv_rows = []
    for model_name, scores in all_models.items():
        results["models"][model_name] = {"categories": {}}
        coverage = model_coverage[model_name]
        print(f"\n--- {model_name} ---")
        print(f"  {'Category':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'ROC':>6} {'PRA':>6}")
        print("  " + "-" * 65)

        for cat in CATEGORIES:
            if cat not in coverage:
                results["models"][model_name]["categories"][cat] = {"note": "N/A — not in model scope"}
                print(f"  {cat:<25} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6}")
                csv_rows.append({
                    "model": model_name, "category": cat, "split": "val",
                    "Precision": "N/A", "Recall": "N/A", "F1": "N/A",
                    "Accuracy": "N/A", "ROC_AUC": "N/A", "PR_AUC": "N/A"
                })
                continue

            m = compute_metrics(y_true[cat], scores[cat])
            results["models"][model_name]["categories"][cat] = m
            if "note" in m:
                print(f"  {cat:<25} -- {m['note']}")
            else:
                roc = f"{m['ROC_AUC']:.3f}" if m["ROC_AUC"] is not None else "N/A"
                pra = f"{m['PR_AUC']:.3f}"  if m["PR_AUC"]  is not None else "N/A"
                print(f"  {cat:<25} {m['Precision']:>6.3f} {m['Recall']:>6.3f} {m['F1']:>6.3f} "
                      f"{m['Accuracy']:>6.3f} {roc:>6} {pra:>6}")
                csv_rows.append({
                    "model": model_name, "category": cat, "split": "val",
                    "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                    "Accuracy": m["Accuracy"], "ROC_AUC": m.get("ROC_AUC"),
                    "PR_AUC": m.get("PR_AUC"), "TP": m["TP"], "FP": m["FP"],
                    "TN": m["TN"], "FN": m["FN"]
                })

    # -------------------------------------------------------------------------
    # Section 9 -- Direct vs Indirect surface breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 9 -- Direct vs Indirect Surface Breakdown (val)")
    print("(Jailbreak and Prompt Injection only)")
    print("=" * 70)

    surfaces_arr = np.array(surfaces)
    direct_idx   = np.where(surfaces_arr == "direct")[0]
    indirect_idx = np.where(surfaces_arr == "indirect")[0]
    print(f"\nVal split: direct={len(direct_idx)}, indirect={len(indirect_idx)}")
    if len(indirect_idx) < 10:
        print(f"  [NOTE] Only {len(indirect_idx)} indirect samples in val — "
              "metrics on indirect surface are low-count and high-variance.")

    surface_results = {}
    for cat in SURFACE_BREAKDOWN_CATS:
        surface_results[cat] = {}
        print(f"\n  Category: {cat}")
        for surf_name, idx in [("direct", direct_idx), ("indirect", indirect_idx)]:
            if len(idx) == 0:
                print(f"    {surf_name}: no samples")
                continue
            for model_name, scores in all_models.items():
                if cat not in model_coverage[model_name]:
                    continue
                y_t = y_true[cat][idx]
                y_p = np.array(scores[cat])[idx]
                m = compute_metrics(y_t, y_p)
                key = f"{model_name}_{surf_name}"
                surface_results[cat][key] = m
                if "note" not in m:
                    print(f"    {model_name:<14} {surf_name:<10}: "
                          f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
                          f"n={len(idx)} pos={m['n_pos']}")

    results["surface_breakdown"] = surface_results

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "benchmark_val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    csv_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model","category","split",
                                                    "Precision","Recall","F1",
                                                    "Accuracy","ROC_AUC","PR_AUC",
                                                    "TP","FP","TN","FN"])
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"CSV summary saved -> {csv_path}")

    print("\n" + "=" * 70)
    print("Phase 2 benchmark complete.")
    print("Next: run phase2_ensemble_ablation.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
