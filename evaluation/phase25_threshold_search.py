"""
Phase 2.5 -- Section 3: Dual-Objective Threshold Calibration.

Replaces Phase 2's single-objective search with TWO separate objectives:

  BLOCK threshold: precision-weighted (auto-reject, irreversible)
    Objective: minimize  0.7*FPR + 0.3*FNR
    Constraint: Recall >= 0.50  (don't sacrifice too much recall)
    Interpretation: false positives (blocking legit traffic) cost 2.3x more
                    than false negatives. Recall floor prevents extreme precision
                    at the cost of completely ignoring attacks.

  REVIEW threshold: recall-weighted (routes to human, reversible)
    Objective: minimize  0.7*FNR + 0.3*FPR  (same as Phase 2 v1)
    Interpretation: false negatives (missing attacks) cost 2.3x more,
                    since a human resolves any false positives.
    Constraint: REVIEW_thr < BLOCK_thr always (REVIEW is looser)

Searched on VALIDATION split only.
Saves: evaluation/results/thresholds_frozen_v2.json
       evaluation/results/threshold_search_v2_val.json
Keeps: evaluation/results/thresholds_frozen.json  (v1, for comparison)

GROUND RULE: data/test.json is never loaded here.
"""

import os, sys, json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np

CATEGORIES   = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR  = "evaluation/results"
RAW_PATH     = os.path.join(RESULTS_DIR, "val_raw_scores.json")
BEST_PATH    = os.path.join(RESULTS_DIR, "best_strategy.json")

SPECIALIST_MAP = {
    "Jailbreak":       "deberta",
    "Prompt Injection":"deberta",
    "PII Leakage":     "presidio",
    "Malicious Tools": None,
    "Hate/Toxicity":   "toxicbert",
}

# Threshold grid
BLOCK_GRID  = np.round(np.arange(0.30, 0.96, 0.05), 2).tolist()
REVIEW_GRID = np.round(np.arange(0.10, 0.71, 0.05), 2).tolist()

# BLOCK objective: precision-weighted
BLOCK_FPR_W  = 0.7   # penalise false positives more
BLOCK_FNR_W  = 0.3
RECALL_FLOOR = 0.50  # minimum recall to keep at block threshold

# REVIEW objective: recall-weighted (same as Phase 2 v1)
REVIEW_FNR_W = 0.7
REVIEW_FPR_W = 0.3


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_ensemble_probs(raw, strategy_name):
    """Reconstruct best-strategy probability scores from val raw scores."""
    if strategy_name.startswith("A_"):
        return {cat: raw["model_a"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("B_"):
        return {cat: raw["model_b"][cat] for cat in CATEGORIES}
    elif strategy_name.startswith("D_"):
        return {cat: [max(a, b) for a, b in zip(raw["model_a"][cat], raw["model_b"][cat])]
                for cat in CATEGORIES}
    elif strategy_name.startswith("G_"):
        import joblib
        meta_clfs = joblib.load(os.path.join(RESULTS_DIR, "meta_classifier_G.pkl"))
        model_keys = ["model_a","model_b","deberta","toxicbert","presidio"]
        probs = {}
        for cat in CATEGORIES:
            if cat in meta_clfs:
                X = np.column_stack([np.array(raw[mk][cat]) for mk in model_keys])
                probs[cat] = list(meta_clfs[cat].predict_proba(X)[:, 1])
            else:
                probs[cat] = raw["ensemble"][cat]
        return probs
    elif strategy_name.startswith("F_"):
        weights = load_json(os.path.join(RESULTS_DIR, "fusion_weights_F.json"))
        probs = {}
        for cat in CATEGORIES:
            w = weights.get(cat, {"w_a": 0.333, "w_b": 0.333, "w_spec": 0.334})
            spec = SPECIALIST_MAP[cat]
            a = np.array(raw["model_a"][cat])
            b = np.array(raw["model_b"][cat])
            s = np.array(raw[spec][cat]) if spec else np.zeros(len(a))
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


def _cm(y_true, y_prob, thr):
    yt = np.array(y_true)
    yp = np.array(y_prob)
    pred = (yp >= thr).astype(int)
    tp = int(((pred==1)&(yt==1)).sum())
    fp = int(((pred==1)&(yt==0)).sum())
    fn = int(((pred==0)&(yt==1)).sum())
    tn = int(((pred==0)&(yt==0)).sum())
    n_pos = int(yt.sum()); n_neg = int(len(yt)-n_pos)
    fnr = fn/max(1, n_pos)
    fpr = fp/max(1, n_neg)
    rec = tp/max(1, n_pos)
    prec = tp/max(1, tp+fp)
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"FNR":fnr,"FPR":fpr,
            "Recall":rec,"Precision":prec,"n_pos":n_pos,"n_neg":n_neg}


def search_block_threshold(y_true, y_prob, label):
    """
    BLOCK: minimize 0.7*FPR + 0.3*FNR subject to Recall >= 0.50.
    If no threshold satisfies the recall floor, relax to best feasible.
    """
    results = []
    for bt in BLOCK_GRID:
        m = _cm(y_true, y_prob, bt)
        if m["n_pos"] == 0:
            continue
        obj = BLOCK_FPR_W * m["FPR"] + BLOCK_FNR_W * m["FNR"]
        results.append({"thr": bt, "obj": round(obj,4), **{k: round(v,4) if isinstance(v,float) else v
                                                            for k,v in m.items()}})

    # First try: satisfies recall floor
    feasible = [r for r in results if r["Recall"] >= RECALL_FLOOR]
    if feasible:
        best = min(feasible, key=lambda r: r["obj"])
        note = f"Recall floor {RECALL_FLOOR} satisfied"
    else:
        # Relax: pick highest recall available
        best = max(results, key=lambda r: r["Recall"]) if results else {"thr": 0.5}
        note = f"Recall floor {RECALL_FLOOR} NOT satisfied; picked highest-recall threshold"
    return best, note, results


def search_review_threshold(y_true, y_prob, block_thr, label):
    """
    REVIEW: minimize 0.7*FNR + 0.3*FPR. Must be < block_thr.
    """
    results = []
    for rt in REVIEW_GRID:
        if rt >= block_thr:
            continue
        m = _cm(y_true, y_prob, rt)
        if m["n_pos"] == 0:
            continue
        obj = REVIEW_FNR_W * m["FNR"] + REVIEW_FPR_W * m["FPR"]
        results.append({"thr": rt, "obj": round(obj,4), **{k: round(v,4) if isinstance(v,float) else v
                                                            for k,v in m.items()}})
    if not results:
        return {"thr": max(0.1, block_thr - 0.1)}, "fallback", []
    best = min(results, key=lambda r: r["obj"])
    return best, "OK", results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run phase2_benchmark.py first.")
        sys.exit(1)

    raw  = load_json(RAW_PATH)
    best = load_json(BEST_PATH) if os.path.exists(BEST_PATH) else {}
    strategy = best.get("best_strategy", "D_LR_MLP_MAX")
    probs    = get_ensemble_probs(raw, strategy)
    y_true   = raw["y_true"]
    surfaces = raw["surfaces"]
    n        = raw["n"]

    print("=" * 75)
    print("CipherGuard Phase 2.5 -- Dual-Objective Threshold Calibration (val)")
    print(f"Strategy: {strategy}  |  n={n}")
    print()
    print(f"BLOCK: min({BLOCK_FPR_W}*FPR + {BLOCK_FNR_W}*FNR)  s.t. Recall >= {RECALL_FLOOR}")
    print(f"       --> precision-weighted, Recall floor {RECALL_FLOOR}")
    print(f"REVIEW: min({REVIEW_FNR_W}*FNR + {REVIEW_FPR_W}*FPR)")
    print(f"       --> recall-weighted  (same as Phase 2 v1)")
    print("=" * 75)

    frozen = {}
    full_search = {}

    # ------------------------------------------------------------------
    print(f"\n{'Category':<25} {'BLK':>5} {'P@BLK':>7} {'R@BLK':>7} {'Obj@BLK':>9}"
          f"  |  {'REV':>5} {'P@REV':>7} {'R@REV':>7} {'Obj@REV':>9}  {'Band':>6}")
    print("-" * 90)

    for cat in CATEGORIES:
        yt = y_true[cat]
        yp = probs[cat]

        blk_best, blk_note, blk_all = search_block_threshold(yt, yp, cat)
        rev_best, rev_note, rev_all = search_review_threshold(yt, yp, blk_best["thr"], cat)

        band = round(blk_best["thr"] - rev_best["thr"], 2)

        frozen[cat] = {
            "block":           blk_best["thr"],
            "block_objective": blk_best["obj"],
            "block_Precision": blk_best.get("Precision"),
            "block_Recall":    blk_best.get("Recall"),
            "block_FPR":       blk_best.get("FPR"),
            "block_FNR":       blk_best.get("FNR"),
            "block_note":      blk_note,
            "review":          rev_best["thr"],
            "review_objective":rev_best["obj"],
            "review_Precision":rev_best.get("Precision"),
            "review_Recall":   rev_best.get("Recall"),
            "review_FPR":      rev_best.get("FPR"),
            "review_FNR":      rev_best.get("FNR"),
            "review_note":     rev_note,
            "band_width":      band,
        }
        full_search[cat] = {"block_grid": blk_all[:10], "review_grid": rev_all[:10]}

        bp = blk_best.get("Precision", 0)
        br = blk_best.get("Recall", 0)
        bo = blk_best.get("obj", 0)
        rp = rev_best.get("Precision", 0)
        rr = rev_best.get("Recall", 0)
        ro = rev_best.get("obj", 0)
        print(f"  {cat:<23} {blk_best['thr']:>5.2f} {bp:>7.3f} {br:>7.3f} {bo:>9.3f}"
              f"  |  {rev_best['thr']:>5.2f} {rp:>7.3f} {rr:>7.3f} {ro:>9.3f}  {band:>6.2f}")

        if "NOT" in blk_note:
            print(f"    [!] {cat} BLOCK: {blk_note}")

    # ------------------------------------------------------------------
    # Per-surface thresholds (Jailbreak and PI, where val n supports it)
    surfaces_arr = np.array(surfaces)
    direct_idx   = np.where(surfaces_arr == "direct")[0]
    indirect_idx = np.where(surfaces_arr == "indirect")[0]

    print(f"\nPer-surface (direct={len(direct_idx)}, indirect={len(indirect_idx)}):")
    SURFACE_CATS = ["Jailbreak", "Prompt Injection"]

    for cat in SURFACE_CATS:
        frozen[cat]["per_surface"] = {}
        for surf_name, idx in [("direct", direct_idx), ("indirect", indirect_idx)]:
            if len(idx) < 5:
                print(f"  {cat}/{surf_name}: too few samples"); continue
            yt_s = [y_true[cat][i] for i in idx]
            yp_s = [probs[cat][i]  for i in idx]
            if sum(yt_s) < 3:
                print(f"  {cat}/{surf_name}: <3 positives -- skipping"); continue

            blk_s, _, _ = search_block_threshold(yt_s, yp_s, f"{cat}/{surf_name}")
            rev_s, _, _ = search_review_threshold(yt_s, yp_s, blk_s["thr"], f"{cat}/{surf_name}")
            frozen[cat]["per_surface"][surf_name] = {
                "block": blk_s["thr"], "review": rev_s["thr"],
                "block_Precision": blk_s.get("Precision"),
                "block_Recall":    blk_s.get("Recall"),
                "n_samples": len(idx), "n_pos": sum(yt_s),
            }
            print(f"  {cat:<25} {surf_name:<10}: "
                  f"BLOCK={blk_s['thr']:.2f} (P={blk_s.get('Precision',0):.3f} "
                  f"R={blk_s.get('Recall',0):.3f}) "
                  f"REVIEW={rev_s['thr']:.2f}  n={len(idx)} pos={sum(yt_s)}")

    # ------------------------------------------------------------------
    search_path = os.path.join(RESULTS_DIR, "threshold_search_v2_val.json")
    with open(search_path, "w", encoding="utf-8") as f:
        json.dump({
            "data_split": "val", "strategy": strategy,
            "block_objective": f"{BLOCK_FPR_W}*FPR + {BLOCK_FNR_W}*FNR (recall_floor={RECALL_FLOOR})",
            "review_objective": f"{REVIEW_FNR_W}*FNR + {REVIEW_FPR_W}*FPR",
            "search": full_search
        }, f, indent=2)
    print(f"\nFull search saved -> {search_path}")

    frozen_path = os.path.join(RESULTS_DIR, "thresholds_frozen_v2.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump({
            "frozen": True,
            "version": "v2",
            "data_split": "val",
            "strategy": strategy,
            "block_objective": f"min({BLOCK_FPR_W}*FPR + {BLOCK_FNR_W}*FNR), Recall >= {RECALL_FLOOR}",
            "review_objective": f"min({REVIEW_FNR_W}*FNR + {REVIEW_FPR_W}*FPR)",
            "note": ("Phase 2.5 dual-objective thresholds. "
                     "BLOCK = precision-weighted (auto-reject). "
                     "REVIEW = recall-weighted (routes to human). "
                     "Supersedes thresholds_frozen.json (v1)."),
            "thresholds": frozen,
        }, f, indent=2)
    print(f"FROZEN v2 thresholds saved -> {frozen_path}")
    print("\n[NOTE] thresholds_frozen.json (v1) preserved for Phase 2 comparison.")

    print("\n" + "=" * 75)
    print("Thresholds FROZEN as v2. Phase 3 must use thresholds_frozen_v2.json.")
    print("Next: run phase25_final_test_eval.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
