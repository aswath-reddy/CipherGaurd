"""
Phase 2 -- Section 11: Ensemble Fusion Ablation (Strategies A through G).

Reads val_raw_scores.json produced by phase2_benchmark.py.
All weight/meta-model learning uses VALIDATION predictions only.
Test set is never touched here.

Strategies:
  A  -- LogReg only
  B  -- MLP only
  C  -- Specialists only (DeBERTa/JB+PI, ToxicBERT/HT, Presidio/PII, 0/MT)
  D  -- max(A, B)
  E  -- max(A, B, specialist)  [current production strategy]
  F  -- weighted average of A, B, specialist  [learned on val]
  G  -- per-category LogReg meta-classifier on stacked val predictions

Outputs:
  evaluation/results/ablation_val_results.json
  evaluation/results/ablation_summary.csv
  evaluation/results/best_strategy.json
  evaluation/results/fusion_weights_F.json
  evaluation/results/meta_classifier_G.pkl
"""

import os, sys, json, csv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

CATEGORIES   = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR  = "evaluation/results"
RAW_PATH     = os.path.join(RESULTS_DIR, "val_raw_scores.json")

# Which specialist covers each category (None = no specialist)
SPECIALIST_MAP = {
    "Jailbreak":       "deberta",
    "Prompt Injection":"deberta",
    "PII Leakage":     "presidio",
    "Malicious Tools": None,
    "Hate/Toxicity":   "toxicbert",
}


def load_raw():
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


def macro_f1(y_true_dict, y_pred_dict):
    scores = []
    for cat in CATEGORIES:
        yt = np.array(y_true_dict[cat])
        yp = np.array(y_pred_dict[cat])
        if yt.sum() == 0:
            continue
        scores.append(f1_score(yt, yp, zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


def per_cat_metrics(y_true_dict, y_prob_dict, threshold=0.5):
    out = {}
    for cat in CATEGORIES:
        yt = np.array(y_true_dict[cat])
        yp = np.array(y_prob_dict[cat])
        pred = (yp >= threshold).astype(int)
        out[cat] = {
            "F1":        round(float(f1_score(yt, pred, zero_division=0)), 4),
            "Precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
            "Recall":    round(float(recall_score(yt, pred, zero_division=0)), 4),
            "Accuracy":  round(float(accuracy_score(yt, pred)), 4),
        }
    f1s = [out[c]["F1"] for c in CATEGORIES if y_true_dict[c].count(1) > 0]
    out["macro_F1"] = round(float(np.mean(f1s)) if f1s else 0.0, 4)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Strategy implementations  (all return: Dict[cat -> list[float]] probabilities)
# ──────────────────────────────────────────────────────────────────────────────

def strategy_A(raw):
    return {cat: raw["model_a"][cat] for cat in CATEGORIES}

def strategy_B(raw):
    return {cat: raw["model_b"][cat] for cat in CATEGORIES}

def strategy_C(raw):
    probs = {}
    for cat in CATEGORIES:
        spec = SPECIALIST_MAP[cat]
        if spec and spec in raw:
            probs[cat] = raw[spec][cat]
        else:
            probs[cat] = [0.0] * len(raw["model_a"][cat])
    return probs

def strategy_D(raw):
    return {cat: [max(a, b) for a, b in zip(raw["model_a"][cat], raw["model_b"][cat])]
            for cat in CATEGORIES}

def strategy_E(raw):
    """max(A, B, specialist) — current production."""
    probs = {}
    for cat in CATEGORIES:
        spec = SPECIALIST_MAP[cat]
        if spec and spec in raw:
            probs[cat] = [max(a, b, s) for a, b, s in
                          zip(raw["model_a"][cat], raw["model_b"][cat], raw[spec][cat])]
        else:
            probs[cat] = [max(a, b) for a, b in
                          zip(raw["model_a"][cat], raw["model_b"][cat])]
    return probs

def strategy_F(raw, y_true):
    """
    Weighted average with weights learned from val predictions via scipy.optimize.
    Weights per category: (w_a, w_b, w_spec) with w_a+w_b+w_spec=1, w_i>=0.
    Objective: maximise F1 on val.
    """
    from scipy.optimize import minimize

    weights_out = {}
    probs_out   = {}

    for cat in CATEGORIES:
        spec = SPECIALIST_MAP[cat]
        a  = np.array(raw["model_a"][cat])
        b  = np.array(raw["model_b"][cat])
        s  = np.array(raw[spec][cat]) if spec and spec in raw else np.zeros(len(a))
        yt = np.array(y_true[cat])

        if yt.sum() == 0:
            # No positives — use equal weights, can't optimise
            probs_out[cat] = list((a + b + s) / 3)
            weights_out[cat] = {"w_a": 0.333, "w_b": 0.333, "w_spec": 0.334}
            continue

        def neg_f1(w):
            wa, wb = w[0], w[1]
            ws = max(0.0, 1.0 - wa - wb)
            yp = wa*a + wb*b + ws*s
            pred = (yp >= 0.5).astype(int)
            return -f1_score(yt, pred, zero_division=0)

        # Constraint: w_a + w_b <= 1; all >= 0
        constraints = [{"type": "ineq", "fun": lambda w: 1.0 - w[0] - w[1]}]
        bounds = [(0, 1), (0, 1)]
        best_res = None
        for x0 in [[0.33, 0.33], [0.5, 0.3], [0.2, 0.2], [0.6, 0.2]]:
            res = minimize(neg_f1, x0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-6, "maxiter": 200})
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        wa = float(max(0, best_res.x[0]))
        wb = float(max(0, best_res.x[1]))
        ws = float(max(0, 1.0 - wa - wb))
        total = wa + wb + ws
        wa, wb, ws = wa/total, wb/total, ws/total

        probs_out[cat]   = list(wa*a + wb*b + ws*s)
        weights_out[cat] = {"w_a": round(wa,4), "w_b": round(wb,4), "w_spec": round(ws,4)}

    return probs_out, weights_out


def strategy_G(raw, y_true):
    """
    Per-category stacking LogReg meta-classifier.
    Features per category: all 5 model scores for that category.
    Evaluated via 5-fold CV on val (to avoid over-optimistic estimate).
    Meta-classifiers also trained on full val for later test-set use.
    """
    model_keys = ["model_a", "model_b", "deberta", "toxicbert", "presidio"]
    meta_clfs  = {}
    probs_out  = {}
    cv_f1s     = {}

    for cat in CATEGORIES:
        yt = np.array(y_true[cat])
        # Feature matrix: n_val x n_models
        X  = np.column_stack([np.array(raw[mk][cat]) for mk in model_keys])

        if yt.sum() < 5:
            # Too few positives for CV
            probs_out[cat] = list(np.mean(X, axis=1))
            cv_f1s[cat]    = None
            continue

        clf = LogisticRegression(C=0.5, max_iter=300, random_state=42,
                                 class_weight="balanced", solver="lbfgs")

        # 5-fold CV on val to estimate unbiased performance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_prob = cross_val_predict(clf, X, yt, cv=cv, method="predict_proba")[:, 1]
        oof_pred = (oof_prob >= 0.5).astype(int)
        cv_f1s[cat] = round(float(f1_score(yt, oof_pred, zero_division=0)), 4)

        # Train on full val for deployment
        clf.fit(X, yt)
        meta_clfs[cat] = clf
        probs_out[cat] = list(clf.predict_proba(X)[:, 1])

    return probs_out, meta_clfs, cv_f1s


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run phase2_benchmark.py first.")
        sys.exit(1)

    raw    = load_raw()
    y_true = raw["y_true"]
    n      = raw["n"]

    print("=" * 70)
    print("CipherGuard Phase 2 -- Ensemble Fusion Ablation (val split)")
    print(f"n={n} | strategies: A B C D E F G")
    print("=" * 70)

    ablation = {}

    # ------------------------------------------------------------------
    # Strategies A – E (no learning on val labels)
    # ------------------------------------------------------------------
    for name, fn in [("A_LogReg", strategy_A),
                     ("B_MLP",    strategy_B),
                     ("C_Specialist", strategy_C),
                     ("D_LR_MLP_MAX", strategy_D),
                     ("E_LR_MLP_Spec_MAX", strategy_E)]:
        probs = fn(raw)
        m = per_cat_metrics(y_true, probs)
        ablation[name] = m
        print(f"\n[{name}]  macro-F1={m['macro_F1']:.4f}")
        print(f"  {'Category':<25} {'F1':>6} {'Prec':>6} {'Rec':>6}")
        for cat in CATEGORIES:
            c = m[cat]
            print(f"  {cat:<25} {c['F1']:>6.3f} {c['Precision']:>6.3f} {c['Recall']:>6.3f}")

    # ------------------------------------------------------------------
    # Strategy F: Weighted average (weights learned on val)
    # ------------------------------------------------------------------
    print("\n[F_WeightedAvg]  (learning weights from val predictions) ...")
    probs_f, weights_f = strategy_F(raw, y_true)
    m_f = per_cat_metrics(y_true, probs_f)
    ablation["F_WeightedAvg"] = m_f
    ablation["F_WeightedAvg"]["learned_weights"] = weights_f
    print(f"  macro-F1={m_f['macro_F1']:.4f}")
    print(f"  {'Category':<25} {'F1':>6} {'w_a':>6} {'w_b':>6} {'w_sp':>6}")
    for cat in CATEGORIES:
        c = m_f[cat]
        w = weights_f.get(cat, {})
        print(f"  {cat:<25} {c['F1']:>6.3f} {w.get('w_a',0):>6.3f} "
              f"{w.get('w_b',0):>6.3f} {w.get('w_spec',0):>6.3f}")

    weights_path = os.path.join(RESULTS_DIR, "fusion_weights_F.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_f, f, indent=2)
    print(f"  Weights saved -> {weights_path}")

    # ------------------------------------------------------------------
    # Strategy G: Meta-classifier stacking (5-fold CV on val)
    # ------------------------------------------------------------------
    print("\n[G_MetaClassifier]  (5-fold CV on val — OOF estimate) ...")
    probs_g, meta_clfs, cv_f1s = strategy_G(raw, y_true)
    m_g = per_cat_metrics(y_true, probs_g)
    # Use CV F1s as the honest estimate for selection (OOF, not train-on-val)
    oof_f1s = [cv_f1s[cat] for cat in CATEGORIES if cv_f1s[cat] is not None]
    m_g["macro_F1_CV"] = round(float(np.mean(oof_f1s)) if oof_f1s else 0.0, 4)
    ablation["G_MetaClassifier"] = m_g
    ablation["G_MetaClassifier"]["cv_f1s"] = cv_f1s
    print(f"  macro-F1 (train-on-val, optimistic)={m_g['macro_F1']:.4f}")
    print(f"  macro-F1 (OOF 5-fold CV, honest)   ={m_g['macro_F1_CV']:.4f}")
    print(f"  [NOTE] Prefer macro_F1_CV for strategy selection — not inflated by memorisation.")
    print(f"  {'Category':<25} {'F1(OOF)':>8} {'F1(train)':>10}")
    for cat in CATEGORIES:
        oof = cv_f1s.get(cat)
        trn = m_g[cat]["F1"]
        print(f"  {cat:<25} {str(round(oof,3)) if oof is not None else 'N/A':>8} {trn:>10.3f}")

    # Save meta-classifiers
    if meta_clfs:
        clf_path = os.path.join(RESULTS_DIR, "meta_classifier_G.pkl")
        joblib.dump(meta_clfs, clf_path)
        print(f"  Meta-classifiers saved -> {clf_path}")

    # ------------------------------------------------------------------
    # Select best strategy (by val macro-F1, using honest estimate for G)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Strategy comparison (macro-F1 on val):")
    print("-" * 70)
    comparison = {}
    for name, m in ablation.items():
        # For G use OOF CV estimate
        score = m.get("macro_F1_CV", m.get("macro_F1", 0.0))
        comparison[name] = score
        marker = " <-- current production" if name == "E_LR_MLP_Spec_MAX" else ""
        print(f"  {name:<25}: {score:.4f}{marker}")

    best_name  = max(comparison, key=comparison.get)
    best_score = comparison[best_name]
    print(f"\nBest strategy: {best_name}  (macro-F1={best_score:.4f})")

    best_strategy = {
        "best_strategy": best_name,
        "macro_F1_val": best_score,
        "selection_basis": "validation macro-F1 (OOF estimate for G)",
        "comparison": comparison,
    }
    best_path = os.path.join(RESULTS_DIR, "best_strategy.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_strategy, f, indent=2)
    print(f"Best strategy saved -> {best_path}")

    # Full ablation results
    abl_path = os.path.join(RESULTS_DIR, "ablation_val_results.json")
    with open(abl_path, "w", encoding="utf-8") as f:
        json.dump({"data_split": "val", "strategies": ablation,
                   "best_strategy": best_name}, f, indent=2)
    print(f"Ablation results saved -> {abl_path}")

    # CSV summary
    csv_rows = []
    for name, m in ablation.items():
        row = {"strategy": name}
        for cat in CATEGORIES:
            row[f"{cat}_F1"] = m[cat]["F1"]
        row["macro_F1"] = m.get("macro_F1_CV", m.get("macro_F1"))
        csv_rows.append(row)
    csv_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Ablation CSV saved -> {csv_path}")

    print("\n" + "=" * 70)
    print("Phase 2 ensemble ablation complete.")
    print("Next: run phase2_threshold_search.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
