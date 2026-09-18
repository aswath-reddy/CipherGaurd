"""
Phase 2 -- Section 12: Threshold Calibration on validation split.

Searches BLOCK and REVIEW threshold combinations per category using a
security-biased objective: minimise 0.7*FNR + 0.3*FPR.

Considers per-surface thresholds for Jailbreak and Prompt Injection
(indirect surface uses slightly lower BLOCK threshold).

Outputs (frozen before any test inspection):
  evaluation/results/threshold_search_val.json
  evaluation/results/thresholds_frozen.json

GROUND RULE: data/test.json is never loaded here.
"""

import os, sys, json, csv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np

CATEGORIES   = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR  = "evaluation/results"
RAW_PATH     = os.path.join(RESULTS_DIR, "val_raw_scores.json")
BEST_PATH    = os.path.join(RESULTS_DIR, "best_strategy.json")

# Threshold search grid
BLOCK_GRID  = np.round(np.arange(0.30, 0.91, 0.05), 2).tolist()
REVIEW_GRID = np.round(np.arange(0.20, 0.71, 0.05), 2).tolist()

# Security objective weight: FNR (missing attacks) is penalised more than FPR
FNR_WEIGHT  = 0.7
FPR_WEIGHT  = 0.3

# Categories where surface-specific thresholds are evaluated
SURFACE_SPLIT_CATS = ["Jailbreak", "Prompt Injection"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_ensemble_probs(raw, strategy_name):
    """Reconstruct the best-strategy probability scores from raw scores."""
    model_keys = ["model_a", "model_b", "deberta", "toxicbert", "presidio"]
    SPECIALIST_MAP = {
        "Jailbreak":       "deberta",
        "Prompt Injection":"deberta",
        "PII Leakage":     "presidio",
        "Malicious Tools": None,
        "Hate/Toxicity":   "toxicbert",
    }
    if strategy_name.startswith("A_"):
        return {cat: raw["model_a"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("B_"):
        return {cat: raw["model_b"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("C_"):
        probs = {}
        for cat in CATEGORIES:
            spec = SPECIALIST_MAP[cat]
            probs[cat] = raw[spec][cat] if spec else [0.0]*len(raw["model_a"][cat])
        return probs
    elif strategy_name.startswith("D_"):
        return {cat: [max(a,b) for a,b in zip(raw["model_a"][cat], raw["model_b"][cat])]
                for cat in CATEGORIES}
    elif strategy_name.startswith("G_"):
        # Load and apply meta-classifier
        import joblib
        clf_path = os.path.join(RESULTS_DIR, "meta_classifier_G.pkl")
        meta_clfs = joblib.load(clf_path)
        probs = {}
        for cat in CATEGORIES:
            if cat in meta_clfs:
                X = np.column_stack([np.array(raw[mk][cat]) for mk in model_keys])
                probs[cat] = list(meta_clfs[cat].predict_proba(X)[:, 1])
            else:
                probs[cat] = raw["ensemble"][cat]
        return probs
    elif strategy_name.startswith("F_"):
        weights_path = os.path.join(RESULTS_DIR, "fusion_weights_F.json")
        weights = load_json(weights_path)
        probs = {}
        for cat in CATEGORIES:
            w = weights.get(cat, {"w_a": 0.333, "w_b": 0.333, "w_spec": 0.334})
            spec = SPECIALIST_MAP[cat]
            a  = np.array(raw["model_a"][cat])
            b  = np.array(raw["model_b"][cat])
            s  = np.array(raw[spec][cat]) if spec else np.zeros(len(a))
            probs[cat] = list(w["w_a"]*a + w["w_b"]*b + w["w_spec"]*s)
        return probs
    else:  # E or fallback
        probs = {}
        for cat in CATEGORIES:
            spec = SPECIALIST_MAP[cat]
            if spec:
                probs[cat] = [max(a,b,s) for a,b,s in
                              zip(raw["model_a"][cat], raw["model_b"][cat], raw[spec][cat])]
            else:
                probs[cat] = [max(a,b) for a,b in
                              zip(raw["model_a"][cat], raw["model_b"][cat])]
        return probs


def objective(y_true, y_prob, block_thr):
    """Security objective: FNR_WEIGHT*FNR + FPR_WEIGHT*FPR at BLOCK threshold."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    pred   = (y_prob >= block_thr).astype(int)
    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0  # degenerate
    fn  = ((pred == 0) & (y_true == 1)).sum()
    fp  = ((pred == 1) & (y_true == 0)).sum()
    fnr = fn / n_pos
    fpr = fp / n_neg
    return FNR_WEIGHT * fnr + FPR_WEIGHT * fpr


def search_thresholds(y_true, y_prob, cat_name, surface_tag="all"):
    """Grid search over BLOCK thresholds; REVIEW = 0.5 * BLOCK."""
    results = []
    for bt in BLOCK_GRID:
        rt = round(bt * 0.6, 2)  # REVIEW is ~60% of BLOCK
        rt = min(rt, bt - 0.05)
        rt = max(rt, 0.10)
        obj = objective(y_true, y_prob, bt)
        pred_block  = (np.array(y_prob) >= bt).astype(int)
        pred_review = ((np.array(y_prob) >= rt) & (np.array(y_prob) < bt)).astype(int)
        tp = int(((pred_block == 1) & (np.array(y_true) == 1)).sum())
        fn = int(((pred_block == 0) & (np.array(y_true) == 1)).sum())
        fp = int(((pred_block == 1) & (np.array(y_true) == 0)).sum())
        results.append({
            "block_thr": bt, "review_thr": rt,
            "objective": round(float(obj), 4),
            "TP": tp, "FN": fn, "FP": fp,
            "FNR": round(fn/max(1,tp+fn), 4),
            "FPR": round(fp/max(1,fp+int(((pred_block==0)&(np.array(y_true)==0)).sum())), 4),
        })
    results.sort(key=lambda r: r["objective"])
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run phase2_benchmark.py first.")
        sys.exit(1)

    raw  = load_json(RAW_PATH)
    best = load_json(BEST_PATH) if os.path.exists(BEST_PATH) else {"best_strategy": "E_LR_MLP_Spec_MAX"}
    strategy_name = best.get("best_strategy", "E_LR_MLP_Spec_MAX")

    print("=" * 70)
    print("CipherGuard Phase 2 -- Threshold Calibration (val split)")
    print(f"Strategy: {strategy_name}")
    print(f"Objective: {FNR_WEIGHT}*FNR + {FPR_WEIGHT}*FPR  (security-biased)")
    print("=" * 70)

    y_true   = raw["y_true"]
    surfaces = raw["surfaces"]
    probs    = get_ensemble_probs(raw, strategy_name)

    n = raw["n"]
    surfaces_arr = np.array(surfaces)
    direct_idx   = np.where(surfaces_arr == "direct")[0]
    indirect_idx = np.where(surfaces_arr == "indirect")[0]

    full_search   = {}
    frozen_thrs   = {}

    print(f"\n{'Category':<25} {'BLOCK':>6} {'REVIEW':>7} {'Obj':>6} {'FNR':>6} {'FPR':>6}")
    print("-" * 70)

    for cat in CATEGORIES:
        yt   = [y_true[cat][i] for i in range(n)]
        yp   = probs[cat]
        res  = search_thresholds(yt, yp, cat)
        full_search[cat] = {"overall": res[:10]}  # top-10 configs
        best_cfg = res[0]

        frozen_thrs[cat] = {
            "block":  best_cfg["block_thr"],
            "review": best_cfg["review_thr"],
            "objective_val": best_cfg["objective"],
            "FNR_val": best_cfg["FNR"],
            "FPR_val": best_cfg["FPR"],
        }
        print(f"  {cat:<23} {best_cfg['block_thr']:>6.2f} {best_cfg['review_thr']:>7.2f} "
              f"{best_cfg['objective']:>6.3f} {best_cfg['FNR']:>6.3f} {best_cfg['FPR']:>6.3f}")

    # Per-surface thresholds for Jailbreak and Prompt Injection
    print(f"\nPer-surface threshold search for: {SURFACE_SPLIT_CATS}")
    print(f"  (indirect val n={len(indirect_idx)} — treat with caution)")

    for cat in SURFACE_SPLIT_CATS:
        full_search[cat]["per_surface"] = {}
        frozen_thrs[cat]["per_surface"] = {}

        for surf_name, idx in [("direct", direct_idx), ("indirect", indirect_idx)]:
            if len(idx) < 5:
                print(f"  {cat} / {surf_name}: too few samples ({len(idx)}) — skipping")
                continue
            yt_s = [y_true[cat][i] for i in idx]
            yp_s = [probs[cat][i]  for i in idx]
            if sum(yt_s) == 0:
                print(f"  {cat} / {surf_name}: no positives — skipping")
                continue
            res_s = search_thresholds(yt_s, yp_s, cat, surf_name)
            full_search[cat]["per_surface"][surf_name] = res_s[:5]
            bc = res_s[0]
            frozen_thrs[cat]["per_surface"][surf_name] = {
                "block":  bc["block_thr"],
                "review": bc["review_thr"],
                "objective_val": bc["objective"],
                "FNR_val": bc["FNR"],
                "FPR_val": bc["FPR"],
                "n_samples": len(idx),
            }
            print(f"  {cat:<25} {surf_name:<10}: "
                  f"BLOCK={bc['block_thr']:.2f} REVIEW={bc['review_thr']:.2f} "
                  f"Obj={bc['objective']:.3f} FNR={bc['FNR']:.3f} FPR={bc['FPR']:.3f}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    search_path = os.path.join(RESULTS_DIR, "threshold_search_val.json")
    with open(search_path, "w", encoding="utf-8") as f:
        json.dump({"data_split": "val", "strategy": strategy_name,
                   "objective_formula": f"{FNR_WEIGHT}*FNR + {FPR_WEIGHT}*FPR",
                   "grid": {"block": BLOCK_GRID, "review": REVIEW_GRID},
                   "results": full_search}, f, indent=2)
    print(f"\nFull search saved -> {search_path}")

    frozen_path = os.path.join(RESULTS_DIR, "thresholds_frozen.json")
    frozen_out  = {
        "frozen": True,
        "data_split": "val",
        "strategy": strategy_name,
        "objective_formula": f"{FNR_WEIGHT}*FNR + {FPR_WEIGHT}*FPR",
        "note": ("Thresholds frozen before test-set inspection. "
                 "Do NOT modify after running phase2_final_test_eval.py."),
        "thresholds": frozen_thrs,
    }
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen_out, f, indent=2)
    print(f"FROZEN thresholds saved -> {frozen_path}")

    print("\n" + "=" * 70)
    print("Thresholds are now FROZEN. Do NOT re-run this script after")
    print("phase2_final_test_eval.py has been run.")
    print("Next: run phase2_final_test_eval.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
