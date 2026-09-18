"""
Dataset Loader and Constructor for CipherGuard Experiments.

DATASET ROLES
─────────────
  pilot_dataset_85.json  → "Pilot / Proof-of-Concept dataset (n=85)".
                           Used for feasibility checks and smoke tests ONLY.
                           NOT a training or evaluation dataset.
                           See data/PILOT_DATASET_README.md.

  expanded_dataset.json  → Full combined dataset (before splitting).
                           Input to split_dataset.py.

  data/train.json        → Training split (70%)
  data/val.json          → Validation split (15%)
  data/test.json         → Final held-out test split (15%)
                           Do NOT tune thresholds on this.

Matches Section IV-A specifications.
"""

import json
import os
from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Pilot dataset — Proof-of-concept (n=85), READ-ONLY
# ─────────────────────────────────────────────────────────────────────────────

def load_pilot_dataset(dataset_path: str = "data/pilot_dataset_85.json") -> List[Dict[str, Any]]:
    """
    Loads the n=85 pilot / proof-of-concept dataset.

    IMPORTANT: This is the Pilot / Proof-of-Concept dataset ONLY (n=85).
    It is NOT used for training or final evaluation. It serves as a quick
    smoke test and feasibility demonstration. See data/PILOT_DATASET_README.md.

    The file is READ-ONLY. build_expanded_dataset.py will abort if it detects
    any modifications to this file (SHA-256 checksum enforced).
    """
    if not os.path.exists(dataset_path):
        possible_paths = [
            dataset_path,
            os.path.join(os.path.dirname(__file__), "..", "data", "pilot_dataset_85.json"),
            os.path.join(os.path.dirname(__file__), "data", "pilot_dataset_85.json"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                dataset_path = p
                break

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_pilot_dataset_readonly(dataset_path: str = "data/pilot_dataset_85.json") -> List[Dict[str, Any]]:
    """
    Alias of load_pilot_dataset() with a more explicit name emphasising the
    read-only / proof-of-concept role of this data.

    Pilot / Proof-of-Concept dataset (n=85). NEVER used for training or
    evaluation. See data/PILOT_DATASET_README.md.
    """
    return load_pilot_dataset(dataset_path)


# ─────────────────────────────────────────────────────────────────────────────
# Expanded dataset (full, pre-split)
# ─────────────────────────────────────────────────────────────────────────────

def load_expanded_dataset(dataset_path: str = "data/expanded_dataset.json") -> List[Dict[str, Any]]:
    """
    Loads the expanded multi-label dataset produced by build_expanded_dataset.py.
    Falls back to pilot dataset if expanded dataset not yet built.

    Use this as input to split_dataset.py to produce train/val/test splits.
    For evaluation experiments, prefer the split files (load_train/val/test_split).
    """
    search_paths = [
        dataset_path,
        os.path.join(os.path.dirname(__file__), "..", "data", "expanded_dataset.json"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    print("[CipherGuard] expanded_dataset.json not found. Run:")
    print("  python -m evaluation.build_expanded_dataset")
    print("[CipherGuard] Falling back to pilot dataset (n=85, proof-of-concept only).")
    return load_pilot_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# Split loaders — use these for training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _load_split(path: str, split_name: str) -> List[Dict[str, Any]]:
    """Internal helper to load a named split file."""
    search_paths = [
        path,
        os.path.join(os.path.dirname(__file__), "..", path),
    ]
    for p in search_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    print(f"[CipherGuard] {split_name} split file not found ({path}).")
    print("  Run: python -m evaluation.split_dataset")
    return []


def load_train_split(path: str = "data/train.json") -> List[Dict[str, Any]]:
    """
    Loads the training split (70% of expanded dataset, seeded split).
    Use for model training and cross-validation.
    """
    return _load_split(path, "train")


def load_val_split(path: str = "data/val.json") -> List[Dict[str, Any]]:
    """
    Loads the validation split (15% of expanded dataset, seeded split).
    Use for hyperparameter tuning and early stopping.
    """
    return _load_split(path, "val")


def load_test_split(path: str = "data/test.json") -> List[Dict[str, Any]]:
    """
    Loads the final held-out test split (15% of expanded dataset, seeded split).

    WARNING: Do NOT use this split for hyperparameter tuning, threshold
    adjustment, or any decision that feeds back into model development.
    Reserve this split for final evaluation reporting only.
    """
    return _load_split(path, "test")


def load_all_splits(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    test_path: str = "data/test.json",
) -> Tuple[List, List, List]:
    """
    Convenience loader returning (train, val, test) as a tuple.
    """
    return (
        load_train_split(train_path),
        load_val_split(val_path),
        load_test_split(test_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Surface splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_by_surface(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits dataset into direct user-turn and indirect context slices.
    """
    direct = [d for d in data if d.get("surface") == "direct"]
    indirect = [d for d in data if d.get("surface") == "indirect"]
    return direct, indirect


# ─────────────────────────────────────────────────────────────────────────────
# Label extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_category_labels(data: List[Dict[str, Any]], category: str) -> List[int]:
    """
    Extracts binary labels for a specific category from a multi-label dataset.
    Handles both old binary format (single 'label') and multi-label format ('labels' dict).
    """
    labels = []
    for d in data:
        if "labels" in d:
            labels.append(int(d["labels"].get(category, 0)))
        else:
            # Legacy binary format: treat label=1 as Jailbreak + Injection
            if category in ("Jailbreak", "Prompt Injection"):
                labels.append(int(d.get("label", 0)))
            else:
                labels.append(0)
    return labels
