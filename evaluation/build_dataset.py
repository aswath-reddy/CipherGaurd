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
