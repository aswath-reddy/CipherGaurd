"""
Dataset Loader and Constructor for CipherGuard Experiments.
Matches Section IV-A specifications.
"""

import json
import os
from typing import List, Dict, Any, Tuple


def load_pilot_dataset(dataset_path: str = "data/pilot_dataset_85.json") -> List[Dict[str, Any]]:
    """
    Loads the n=85 benchmark pilot dataset.
    """
    if not os.path.exists(dataset_path):
        # Fallback to absolute or parent search
        possible_paths = [
            dataset_path,
            os.path.join(os.path.dirname(__file__), "..", "data", "pilot_dataset_85.json"),
            os.path.join(os.path.dirname(__file__), "data", "pilot_dataset_85.json")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                dataset_path = p
                break

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_by_surface(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits dataset into direct user-turn and indirect context slices.
    """
    direct = [d for d in data if d.get("surface") == "direct"]
    indirect = [d for d in data if d.get("surface") == "indirect"]
    return direct, indirect


def load_expanded_dataset(dataset_path: str = "data/expanded_dataset.json") -> List[Dict[str, Any]]:
    """
    Loads the expanded multi-label dataset built by build_expanded_dataset.py.
    Falls back to pilot dataset if expanded dataset not yet built.
    """
    search_paths = [
        dataset_path,
        os.path.join(os.path.dirname(__file__), "..", "data", "expanded_dataset.json"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    print("[CipherGuard] expanded_dataset.json not found. Run: python -m evaluation.build_expanded_dataset")
    print("[CipherGuard] Falling back to pilot dataset.")
    return load_pilot_dataset()


def get_category_labels(data: List[Dict[str, Any]], category: str) -> List[int]:
    """
    Extracts binary labels for a specific category from a multi-label dataset.
    Handles both old binary format (single 'label') and new multi-label format ('labels' dict).
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
