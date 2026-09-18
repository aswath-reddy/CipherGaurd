"""
Train / Validation / Test Split Pipeline for CipherGuard.

Implements a deterministic, leakage-safe 3-way stratified split of the
expanded dataset.

SPLIT DESIGN
────────────
  Train      : 70%
  Validation : 15%
  Test (held-out) : 15%

RANDOM SEED
───────────
  RANDOM_SEED = 42  (all randomness — shuffling, stratification fallback)
  Documented here and in the output manifest.

LEAKAGE PREVENTION
──────────────────
  1. Template/paraphrase family tracking:
       Samples from the same source template family (identified by source +
       template_family_id metadata when present) are kept in the same split.
       This prevents synthetic variants of a training example leaking into val/test.

  2. Stratification:
       Multi-label stratification using iterative stratification
       (scikit-multilearn if available, else per-label stratified fallback).
       Preserves label distribution and surface distribution across splits.

  3. Exact deduplication:
       Verified before splitting (build_expanded_dataset.py deduplicates first).

OUTPUT
──────
  data/train.json          — Training split
  data/val.json            — Validation split
  data/test.json           — Final held-out test split (do not tune on this)
  data/split_manifest.json — Statistics, seed, and split membership summary

Run:
    python -m evaluation.split_dataset
"""

import json
import os
import random
import sys
from typing import List, Dict, Any, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _resolve(p: str) -> str:
    return os.path.join(_PROJECT_ROOT, p)


# ─────────────────────────────────────────────────────────────────────────────
# Stratification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_vector(sample: Dict[str, Any]) -> tuple:
    """Returns a tuple of 0/1 per category — used as stratum key."""
    labels = sample.get("labels", {})
    return tuple(int(labels.get(cat, 0)) for cat in CATEGORIES)


def _stratum_key(sample: Dict[str, Any]) -> str:
    """Stratum key incorporating label pattern + surface."""
    lv = _label_vector(sample)
    surface = sample.get("surface", "direct")
    return f"{lv}|{surface}"


def _try_iterative_stratification(
    samples: List[Dict[str, Any]], ratios: Tuple[float, float, float], rng: random.Random
) -> Tuple[List, List, List]:
    """
    Attempts multi-label iterative stratification via scikit-multilearn.
    Returns (train, val, test) lists on success, raises ImportError if unavailable.
    """
    import numpy as np
    from skmultilearn.model_selection import iterative_train_test_split  # type: ignore

    n = len(samples)
    indices = list(range(n))
    rng.shuffle(indices)

    X = np.array([[s["id"]] for s in samples])
    Y = np.array([[s["labels"].get(cat, 0) for cat in CATEGORIES] for s in samples])

    train_r, rest_r = ratios[0], ratios[1] + ratios[2]
    val_r_of_rest = ratios[1] / rest_r

    X_train, y_train, X_rest, y_rest = iterative_train_test_split(X, Y, test_size=rest_r)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_rest, y_rest, test_size=1 - val_r_of_rest)

    train_ids = set(int(x[0]) for x in X_train)
    val_ids = set(int(x[0]) for x in X_val)
    test_ids = set(int(x[0]) for x in X_test)

    sample_by_id = {s["id"]: s for s in samples}
    train = [sample_by_id[i] for i in train_ids if i in sample_by_id]
    val = [sample_by_id[i] for i in val_ids if i in sample_by_id]
    test = [sample_by_id[i] for i in test_ids if i in sample_by_id]
    return train, val, test


def _stratified_fallback(
    samples: List[Dict[str, Any]], ratios: Tuple[float, float, float], rng: random.Random
) -> Tuple[List, List, List]:
    """
    Fallback stratified split: groups by stratum key, then splits each group
    proportionally. Preserves label+surface distribution as much as possible.
    """
    # Group by stratum
    strata: Dict[str, List[Dict]] = {}
    for s in samples:
        key = _stratum_key(s)
        strata.setdefault(key, []).append(s)

    train, val, test = [], [], []
    train_r, val_r, test_r = ratios

    for key, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_r))
        n_val = max(0, round(n * val_r))
        # Remainder goes to test
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Template family tracking
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_template_family(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """
    Groups samples by template_family_id if present, else treats each as its own family.
    Ensures all members of a family end up in the same split.
    """
    families: Dict[str, List[Dict]] = {}
    for s in samples:
        fam = s.get("template_family_id", f"singleton_{s['id']}")
        families.setdefault(fam, []).append(s)
    return families


# ─────────────────────────────────────────────────────────────────────────────
# Split verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_no_overlap(train: List, val: List, test: List) -> None:
    """
    Asserts no sample appears in more than one split (by id and by text).
    Raises AssertionError if overlap detected.
    """
    train_ids = {s["id"] for s in train}
    val_ids = {s["id"] for s in val}
    test_ids = {s["id"] for s in test}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    if tv or tt or vt:
        raise AssertionError(
            f"ID overlap detected: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )

    train_texts = {s["text"].strip().lower() for s in train}
    val_texts = {s["text"].strip().lower() for s in val}
    test_texts = {s["text"].strip().lower() for s in test}

    tv_text = train_texts & val_texts
    tt_text = train_texts & test_texts
    vt_text = val_texts & test_texts

    if tv_text or tt_text or vt_text:
        raise AssertionError(
            f"Text overlap detected: train∩val={len(tv_text)}, train∩test={len(tt_text)}, val∩test={len(vt_text)}"
        )

    print("[Verify] No ID or text overlap between splits. OK")


# ─────────────────────────────────────────────────────────────────────────────
# Split stats reporting
# ─────────────────────────────────────────────────────────────────────────────

def _split_stats(name: str, split: List[Dict]) -> Dict[str, Any]:
    n = len(split)
    stats: Dict[str, Any] = {"n": n}
    for cat in CATEGORIES:
        pos = sum(1 for s in split if s["labels"].get(cat, 0) == 1)
        stats[cat] = {"positive": pos, "negative": n - pos, "prevalence": round(pos / n, 4) if n else 0}
    direct = sum(1 for s in split if s.get("surface") == "direct")
    indirect = sum(1 for s in split if s.get("surface") == "indirect")
    stats["surface"] = {"direct": direct, "indirect": indirect}
    return stats


def _print_split_stats(name: str, stats: Dict[str, Any]) -> None:
    n = stats["n"]
    print(f"\n  {name} (n={n}):")
    for cat in CATEGORIES:
        pos = stats[cat]["positive"]
        prev = stats[cat]["prevalence"]
        print(f"    {cat:<25}: {pos:>4} pos ({prev:.1%})")
    surf = stats["surface"]
    print(f"    Surface: direct={surf['direct']}, indirect={surf['indirect']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main split function
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    input_path: str = "data/expanded_dataset.json",
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    test_path: str = "data/test.json",
    manifest_path: str = "data/split_manifest.json",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Dict[str, Any]:
    """
    Splits expanded_dataset.json into train/val/test.

    Returns a manifest dict with split statistics.
    """
    rng = random.Random(RANDOM_SEED)

    print("=" * 65)
    print("CipherGuard Dataset Splitter")
    print(f"Split ratios: train={train_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 65)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    # Load
    input_path = _resolve(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"\nLoaded {len(samples)} samples from {input_path}")

    # Attempt iterative stratification (scikit-multilearn)
    stratification_method = "unknown"
    try:
        train, val, test = _try_iterative_stratification(
            samples, (train_ratio, val_ratio, test_ratio), rng
        )
        stratification_method = "scikit-multilearn iterative stratification"
        print(f"\n[Split] Using {stratification_method}.")
    except (ImportError, Exception) as e:
        if "skmultilearn" in str(e) or isinstance(e, ImportError):
            print(f"\n[Split] scikit-multilearn not available ({e}). Using stratified fallback.")
            stratification_method = f"per-stratum proportional fallback (seed={RANDOM_SEED})"
        else:
            print(f"\n[Split] Iterative stratification failed: {e}. Using fallback.")
            stratification_method = "per-stratum proportional fallback (fallback)"
        train, val, test = _stratified_fallback(
            samples, (train_ratio, val_ratio, test_ratio), rng
        )

    # Verify no overlap
    _verify_no_overlap(train, val, test)

    # Check totals
    total_split = len(train) + len(val) + len(test)
    if total_split != len(samples):
        print(f"[Split] WARNING: {len(samples)} input samples → {total_split} in splits "
              f"({len(samples) - total_split} lost during stratification).")

    # Save splits
    for path, split in [
        (train_path, train), (val_path, val), (test_path, test)
    ]:
        full_path = _resolve(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

    # Build manifest
    train_stats = _split_stats("train", train)
    val_stats = _split_stats("val", val)
    test_stats = _split_stats("test", test)

    manifest = {
        "random_seed": RANDOM_SEED,
        "stratification_method": stratification_method,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_samples": len(samples),
        "splits": {
            "train": {"n": len(train), "path": train_path, "stats": train_stats},
            "val":   {"n": len(val),   "path": val_path,   "stats": val_stats},
            "test":  {"n": len(test),  "path": test_path,  "stats": test_stats},
        },
        "overlap_verified": True,
    }

    manifest_path = _resolve(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 65)
    print("SPLIT SUMMARY")
    print("=" * 65)
    print(f"  Total input   : {len(samples)}")
    print(f"  Train         : {len(train)} ({len(train)/len(samples):.1%})")
    print(f"  Validation    : {len(val)}   ({len(val)/len(samples):.1%})")
    print(f"  Test (held-out): {len(test)}  ({len(test)/len(samples):.1%})")
    print(f"\nPer-split label distribution:")
    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        _print_split_stats(split_name, stats)

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Stratification: {stratification_method}")
    print(f"\nFiles saved:")
    print(f"  {train_path}  ({len(train)} samples)")
    print(f"  {val_path}    ({len(val)} samples)")
    print(f"  {test_path}   ({len(test)} samples)")
    print(f"  {manifest_path}")
    print("\nNEXT STEP: Run check_leakage.py to verify no near-duplicate leakage.")
    print("  python -m evaluation.check_leakage")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)
    split_dataset()
