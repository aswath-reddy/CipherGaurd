"""
Lexical Overlap / Data Leakage Checker for CipherGuard.

Checks for near-duplicate samples between train/val/test splits using
character-level 3-gram Jaccard similarity.

PURPOSE
───────
Section 19 of the dataset spec requires that we detect and report any
near-duplicate overlap between splits to avoid inflated evaluation metrics
caused by lexical/template memorisation.

METHODOLOGY
───────────
  1. For each sample in each split, compute the set of character trigrams.
  2. Compute Jaccard similarity between all cross-split pairs:
       J(A, B) = |A ∩ B| / |A ∪ B|
  3. Report:
       - Maximum pairwise similarity
       - Mean pairwise similarity (sampled for large datasets)
       - Count of pairs above threshold (default 0.80)
       - Whether any exact duplicates exist (J = 1.0)
  4. Save full report to data/leakage_report.json

PERFORMANCE
───────────
  Exhaustive O(n²) comparison is prohibitive for large splits. This script
  uses approximate sampling: up to MAX_PAIRS cross-pairs are sampled at
  random and the worst-case is reported. For exact duplicate detection,
  a hash set comparison is always run (O(n)).

Run:
    python -m evaluation.check_leakage
"""

import json
import os
import random
import sys
from typing import List, Dict, Any, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
SIMILARITY_THRESHOLD: float = 0.80   # pairs above this are "suspicious"
MAX_PAIRS_PER_COMPARISON: int = 50_000  # max cross-pairs to sample per pair of splits
NGRAM_SIZE: int = 3                  # character n-gram size

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _resolve(p: str) -> str:
    return os.path.join(_PROJECT_ROOT, p)


# ─────────────────────────────────────────────────────────────────────────────
# Trigram computation
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(text: str, n: int = NGRAM_SIZE) -> Set[str]:
    """Computes character n-gram set for a string (lowercased, whitespace-normalised)."""
    t = " ".join(text.lower().split())
    if len(t) < n:
        return {t}
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Exact duplicate detection (fast)
# ─────────────────────────────────────────────────────────────────────────────

def check_exact_duplicates(
    split_a: List[Dict], split_b: List[Dict], name_a: str, name_b: str
) -> int:
    """
    Returns count of exact-text duplicates between two splits.
    """
    texts_a = {s["text"].strip().lower() for s in split_a}
    texts_b = {s["text"].strip().lower() for s in split_b}
    exact = texts_a & texts_b
    if exact:
        print(f"  [EXACT DUPLICATE] {name_a}∩{name_b}: {len(exact)} exact text matches!")
    return len(exact)


# ─────────────────────────────────────────────────────────────────────────────
# Near-duplicate detection (sampled)
# ─────────────────────────────────────────────────────────────────────────────

def check_near_duplicates(
    split_a: List[Dict],
    split_b: List[Dict],
    name_a: str,
    name_b: str,
    rng: random.Random,
    threshold: float = SIMILARITY_THRESHOLD,
    max_pairs: int = MAX_PAIRS_PER_COMPARISON,
) -> Dict[str, Any]:
    """
    Samples up to max_pairs cross-pairs from split_a × split_b and computes
    Jaccard similarity on character trigrams.

    Returns a dict with: max_similarity, mean_similarity, pairs_above_threshold,
    sample_size, is_exhaustive.
    """
    ngrams_a = [(s["id"], _ngrams(s["text"])) for s in split_a]
    ngrams_b = [(s["id"], _ngrams(s["text"])) for s in split_b]

    total_pairs = len(ngrams_a) * len(ngrams_b)
    is_exhaustive = total_pairs <= max_pairs

    if is_exhaustive:
        pairs = [(a, b) for a in ngrams_a for b in ngrams_b]
    else:
        # Sample uniformly from the cross-product
        pairs = []
        for _ in range(max_pairs):
            a = rng.choice(ngrams_a)
            b = rng.choice(ngrams_b)
            pairs.append((a, b))

    sims = [_jaccard(a[1], b[1]) for a, b in pairs]

    max_sim = max(sims) if sims else 0.0
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    above = sum(1 for s in sims if s >= threshold)

    print(f"  {name_a}x{name_b}: max={max_sim:.4f}, mean={mean_sim:.4f}, "
          f"pairs>={threshold:.2f}: {above}/{len(sims)} "
          f"({'exhaustive' if is_exhaustive else f'sampled {len(sims)}/{total_pairs}'})")

    if above > 0:
        print(f"  [WARNING] {above} near-duplicate pair(s) found above threshold {threshold}.")

    return {
        "max_similarity": round(max_sim, 6),
        "mean_similarity": round(mean_sim, 6),
        "pairs_above_threshold": above,
        "sample_size": len(sims),
        "total_pairs": total_pairs,
        "is_exhaustive": is_exhaustive,
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main check
# ─────────────────────────────────────────────────────────────────────────────

def run_leakage_check(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    test_path: str = "data/test.json",
    report_path: str = "data/leakage_report.json",
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Runs the full leakage check across all split pairs and saves a report.
    """
    rng = random.Random(RANDOM_SEED)

    print("=" * 65)
    print("CipherGuard Dataset Leakage Checker")
    print(f"Jaccard threshold: {threshold} | N-gram size: {NGRAM_SIZE} chars")
    print(f"Max sampled pairs per comparison: {MAX_PAIRS_PER_COMPARISON:,}")
    print("=" * 65)

    # Load splits
    splits = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        full_path = _resolve(path)
        if not os.path.exists(full_path):
            print(f"[SKIP] {full_path} does not exist. Run split_dataset.py first.")
            splits[name] = []
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                splits[name] = json.load(f)
            print(f"Loaded {name}: {len(splits[name])} samples")

    if not all(splits.values()):
        print("\n[ERROR] One or more splits missing. Cannot run leakage check.")
        return {}

    print("\n[1/3] Checking exact duplicates...")
    exact_tv = check_exact_duplicates(splits["train"], splits["val"], "train", "val")
    exact_tt = check_exact_duplicates(splits["train"], splits["test"], "train", "test")
    exact_vt = check_exact_duplicates(splits["val"], splits["test"], "val", "test")

    total_exact = exact_tv + exact_tt + exact_vt
    if total_exact == 0:
        print("  No exact duplicates across splits. OK")

    print("\n[2/3] Checking near-duplicates (Jaccard trigram similarity)...")
    nd_tv = check_near_duplicates(splits["train"], splits["val"],   "train", "val",  rng, threshold)
    nd_tt = check_near_duplicates(splits["train"], splits["test"],  "train", "test", rng, threshold)
    nd_vt = check_near_duplicates(splits["val"],   splits["test"],  "val",   "test", rng, threshold)

    # Worst-case summary
    max_overall = max(nd_tv["max_similarity"], nd_tt["max_similarity"], nd_vt["max_similarity"])
    total_suspicious = nd_tv["pairs_above_threshold"] + nd_tt["pairs_above_threshold"] + nd_vt["pairs_above_threshold"]

    print("\n[3/3] Summary")
    print(f"  Exact duplicates across splits : {total_exact}")
    print(f"  Max Jaccard similarity         : {max_overall:.4f}")
    print(f"  Suspicious pairs (>={threshold}) : {total_suspicious}")
    if total_exact == 0 and total_suspicious == 0:
        print("  Result: NO LEAKAGE DETECTED - CLEAN")
    elif total_exact > 0:
        print("  Result: EXACT DUPLICATE LEAKAGE -- fix before training!")
    else:
        print(f"  Result: {total_suspicious} near-duplicate pairs warrant manual review.")

    # Build and save report
    report = {
        "random_seed": RANDOM_SEED,
        "ngram_size": NGRAM_SIZE,
        "threshold": threshold,
        "max_pairs_sampled": MAX_PAIRS_PER_COMPARISON,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "exact_duplicates": {
            "train_val": exact_tv,
            "train_test": exact_tt,
            "val_test": exact_vt,
            "total": total_exact,
        },
        "near_duplicates": {
            "train_val": nd_tv,
            "train_test": nd_tt,
            "val_test": nd_vt,
        },
        "max_jaccard_overall": round(max_overall, 6),
        "total_suspicious_pairs": total_suspicious,
        "verdict": (
            "EXACT_DUPLICATES" if total_exact > 0
            else ("NEAR_DUPLICATES_FOUND" if total_suspicious > 0 else "CLEAN")
        ),
    }

    report_path = _resolve(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)
    run_leakage_check()
