"""
Phase 3 - Section 13: Calibration Analysis.

Evaluates whether Strategy D risk scores are well-calibrated.
Computes reliability curves, Brier score, and ECE before and after:
  - Platt scaling (logistic sigmoid fit on val probabilities)
  - Isotonic regression fit on val probabilities

Calibration is FITTED on validation data only.
Test-set calibrated metrics are reported as a secondary check
(thresholds are not re-tuned on test results).

Outputs:
  evaluation/results/calibration_results.json
  evaluation/results/calibration_curves.json  (reliability plot data)
"""

import os, sys, json, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

CATEGORIES  = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
RESULTS_DIR = "evaluation/results"
RAW_PATH    = os.path.join(RESULTS_DIR, "val_raw_scores.json")
BEST_PATH   = os.path.join(RESULTS_DIR, "best_strategy.json")
N_BINS      = 10

# ── helpers ──────────────────────────────────────────────────────────────────

def strategy_d_probs(raw):
    return {cat: [max(a, b) for a, b in zip(raw["model_a"][cat], raw["model_b"][cat])]
            for cat in CATEGORIES}


def ece(y_true, y_prob, n_bins=10):
    yt = np.array(y_true); yp = np.array(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    bin_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (yp >= lo) & (yp < hi)
        if lo == bins[-2]:          # last bin: include right edge
            mask = (yp >= lo) & (yp <= hi)
        if mask.sum() == 0:
            bin_data.append({"bin_lo": round(float(lo),2), "bin_hi": round(float(hi),2),
                             "n": 0, "mean_pred": None, "frac_pos": None, "gap": None})
            continue
        mean_pred = float(yp[mask].mean())
        frac_pos  = float(yt[mask].mean())
        gap       = abs(mean_pred - frac_pos)
        weight    = mask.sum() / len(yt)
        ece_val  += weight * gap
        bin_data.append({"bin_lo": round(float(lo),2), "bin_hi": round(float(hi),2),
                         "n": int(mask.sum()), "mean_pred": round(mean_pred,4),
                         "frac_pos": round(frac_pos,4), "gap": round(gap,4)})
    return round(ece_val, 5), bin_data


def fit_platt(y_true, y_prob):
    """Fit Platt scaling: logistic regression with y_prob as single feature."""
    X = np.array(y_prob).reshape(-1, 1)
    y = np.array(y_true)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(X, y)
    return clf


def fit_isotonic(y_true, y_prob):
    """Fit isotonic regression calibration."""
    y = np.array(y_true)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y)
    return iso


def apply_platt(clf, y_prob):
    if clf is None: return y_prob
    return list(clf.predict_proba(np.array(y_prob).reshape(-1,1))[:,1])


def apply_isotonic(iso, y_prob):
    if iso is None: return y_prob
    return list(iso.predict(y_prob))


# ── load val data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("CipherGuard Phase 3 -- Section 13: Calibration Analysis")
print("=" * 70)

raw = json.load(open(RAW_PATH, encoding="utf-8"))
probs_d = strategy_d_probs(raw)
y_true_val = raw["y_true"]
n_val = raw["n"]
print(f"Val split: n={n_val}")

# ── load test data for secondary check ───────────────────────────────────────
print("Loading test split for secondary calibration check ...")
test_data = json.load(open("data/test.json", encoding="utf-8"))
texts_t  = [d["text"] for d in test_data]
surfaces_t = [d.get("surface", "direct") for d in test_data]
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from evaluation.build_dataset import get_category_labels

mA = ModelFamilyA.load("models")
mB = ModelFamilyB.load("models")
batch_a = mA.score_batch(texts_t, surfaces_t)
batch_b = mB.score_batch(texts_t, surfaces_t)
probs_test = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a, batch_b)]
              for cat in CATEGORIES}
y_true_test = {cat: get_category_labels(test_data, cat) for cat in CATEGORIES}
print(f"Test split: n={len(test_data)}")

# ── calibration analysis ──────────────────────────────────────────────────────
print()
print(f"{'Category':<25} {'Brier':>7} {'ECE':>7} {'Brier_P':>8} {'ECE_P':>7} {'Brier_I':>8} {'ECE_I':>7}")
print("-" * 75)

results = {}
curves  = {}

platts   = {}
isotonics = {}

for cat in CATEGORIES:
    yt_v  = y_true_val[cat]
    yp_v  = probs_d[cat]

    # Before calibration
    brier_raw  = round(float(brier_score_loss(yt_v, yp_v)), 5)
    ece_raw, bins_raw = ece(yt_v, yp_v)

    # Fit calibrators on val
    platt_clf = fit_platt(yt_v, yp_v)
    iso_clf   = fit_isotonic(yt_v, yp_v)
    platts[cat]   = platt_clf
    isotonics[cat] = iso_clf

    # Val calibrated scores
    yp_v_platt = apply_platt(platt_clf, yp_v)
    yp_v_iso   = apply_isotonic(iso_clf, yp_v)

    brier_platt = round(float(brier_score_loss(yt_v, yp_v_platt)), 5) if platt_clf else None
    ece_platt, bins_platt = ece(yt_v, yp_v_platt) if platt_clf else (None, [])
    brier_iso   = round(float(brier_score_loss(yt_v, yp_v_iso)), 5) if iso_clf else None
    ece_iso, bins_iso = ece(yt_v, yp_v_iso) if iso_clf else (None, [])

    # Val after calibration (using isotonic since it's non-parametric)
    brier_best = brier_iso if brier_iso and brier_iso < brier_raw else brier_raw
    ece_best   = ece_iso   if ece_iso   and ece_iso   < ece_raw   else ece_raw

    # Test calibrated scores
    yt_t  = y_true_test[cat]
    yp_t  = probs_test[cat]
    yp_t_platt = apply_platt(platt_clf, yp_t)
    yp_t_iso   = apply_isotonic(iso_clf, yp_t)
    brier_test_raw   = round(float(brier_score_loss(yt_t, yp_t)), 5)
    brier_test_platt = round(float(brier_score_loss(yt_t, yp_t_platt)), 5) if platt_clf else None
    brier_test_iso   = round(float(brier_score_loss(yt_t, yp_t_iso)), 5) if iso_clf else None
    ece_test_raw, _  = ece(yt_t, yp_t)
    ece_test_platt, _ = ece(yt_t, yp_t_platt) if platt_clf else (None, [])
    ece_test_iso, _   = ece(yt_t, yp_t_iso) if iso_clf else (None, [])

    print(f"  {cat:<23} {brier_raw:>7.5f} {ece_raw:>7.5f} "
          f"{brier_platt or 0:>8.5f} {ece_platt or 0:>7.5f} "
          f"{brier_iso or 0:>8.5f} {ece_iso or 0:>7.5f}")

    results[cat] = {
        "val": {
            "n_pos": int(sum(yt_v)),
            "brier_raw": brier_raw,  "ece_raw": ece_raw,
            "brier_platt": brier_platt, "ece_platt": ece_platt,
            "brier_isotonic": brier_iso, "ece_isotonic": ece_iso,
        },
        "test_secondary": {
            "brier_raw": brier_test_raw, "ece_raw": ece_test_raw,
            "brier_platt": brier_test_platt, "ece_platt": ece_test_platt,
            "brier_isotonic": brier_test_iso, "ece_isotonic": ece_test_iso,
        }
    }
    curves[cat] = {
        "raw": bins_raw,
        "platt": bins_platt,
        "isotonic": bins_iso,
    }

print()
print("  Columns: Brier_raw | ECE_raw | Brier_Platt | ECE_Platt | Brier_Isotonic | ECE_Isotonic")

# ── summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: val calibration (fitted and evaluated on val -- optimistic)")
print("=" * 70)
all_raw_brier   = [results[c]["val"]["brier_raw"] for c in CATEGORIES]
all_platt_brier = [results[c]["val"]["brier_platt"] or results[c]["val"]["brier_raw"] for c in CATEGORIES]
all_iso_brier   = [results[c]["val"]["brier_isotonic"] or results[c]["val"]["brier_raw"] for c in CATEGORIES]
all_raw_ece     = [results[c]["val"]["ece_raw"] for c in CATEGORIES]
all_iso_ece     = [results[c]["val"]["ece_isotonic"] or results[c]["val"]["ece_raw"] for c in CATEGORIES]
print(f"  Mean Brier raw:      {np.mean(all_raw_brier):.5f}")
print(f"  Mean Brier Platt:    {np.mean(all_platt_brier):.5f}")
print(f"  Mean Brier Isotonic: {np.mean(all_iso_brier):.5f}")
print(f"  Mean ECE raw:        {np.mean(all_raw_ece):.5f}")
print(f"  Mean ECE Isotonic:   {np.mean(all_iso_ece):.5f}")

print("\nSECONDARY: test-set calibration metrics (calibrators fitted on val)")
print("-" * 50)
t_raw  = [results[c]["test_secondary"]["brier_raw"] for c in CATEGORIES]
t_iso  = [results[c]["test_secondary"]["brier_isotonic"] or results[c]["test_secondary"]["brier_raw"] for c in CATEGORIES]
t_ece  = [results[c]["test_secondary"]["ece_raw"] for c in CATEGORIES]
t_ece_iso = [results[c]["test_secondary"]["ece_isotonic"] or results[c]["test_secondary"]["ece_raw"] for c in CATEGORIES]
print(f"  Mean Brier raw:      {np.mean(t_raw):.5f}")
print(f"  Mean Brier Isotonic: {np.mean(t_iso):.5f}")
print(f"  Mean ECE raw:        {np.mean(t_ece):.5f}")
print(f"  Mean ECE Isotonic:   {np.mean(t_ece_iso):.5f}")

print("\n[NOTE] Val metrics are optimistic (fit+eval on same set).")
print("[NOTE] Test metrics use val-fitted calibrators -- no threshold re-tuning.")

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "calibration_results.json"), "w", encoding="utf-8") as f:
    json.dump({"timestamp": datetime.datetime.now().isoformat(),
               "strategy": "D_LR_MLP_MAX", "n_bins": N_BINS, "categories": results}, f, indent=2)
with open(os.path.join(RESULTS_DIR, "calibration_curves.json"), "w", encoding="utf-8") as f:
    json.dump(curves, f, indent=2)
print("\nSaved -> evaluation/results/calibration_results.json")
print("Saved -> evaluation/results/calibration_curves.json")
print("\nSection 13 DONE")
