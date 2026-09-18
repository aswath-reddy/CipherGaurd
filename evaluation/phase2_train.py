"""
Phase 2 -- Section 7/8 Prerequisite: Retrain Family A (LogReg) and Family B (MLP)
on data/train.json ONLY (645 samples, seed=42).

Existing models/family_a_*.pkl and models/family_b_*.pkl were trained on all 923
samples (including val+test). This script corrects that by retraining on train only.

GROUND RULE: data/test.json is never loaded or inspected here.
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from evaluation.build_dataset import get_category_labels
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
MODELS_DIR = "models"
TRAIN_PATH = "data/train.json"


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("CipherGuard Phase 2 -- Model Training (train split only)")
    print(f"Input: {TRAIN_PATH}")
    print("=" * 70)

    # -- Load train split only --
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: {TRAIN_PATH} not found. Run split_dataset.py first.")
        sys.exit(1)

    train_data = load_split(TRAIN_PATH)
    texts = [d["text"] for d in train_data]
    print(f"\nTrain split loaded: {len(train_data)} samples")

    print("\nClass distribution (train split):")
    labels_dict = {}
    for cat in CATEGORIES:
        y = get_category_labels(train_data, cat)
        labels_dict[cat] = y
        pos = sum(y)
        print(f"  {cat:<25}: {pos:>3} positive / {len(y) - pos:>3} negative  ({100*pos/len(y):.1f}%)")

    # -------------------------------------------------------------------------
    # Train Family A: TF-IDF + Logistic Regression
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training Model Family A  (TF-IDF + Logistic Regression) ...")
    print("-" * 70)

    model_a = ModelFamilyA(categories=CATEGORIES)
    model_a.fit(texts, labels_dict)

    print("\nTrain-split 5-fold CV F1 (Family A):")
    for cat in CATEGORIES:
        y = np.array(labels_dict[cat])
        if sum(y) < 5:
            print(f"  {cat:<25}: SKIPPED (fewer than 5 positives)")
            continue
        pipe = model_a.pipelines[cat]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, texts, y, cv=cv, scoring="f1", error_score=0.0)
        print(f"  {cat:<25}: F1 = {np.mean(scores):.3f}  (+/-{np.std(scores):.3f})")

    # -------------------------------------------------------------------------
    # Train Family B: TF-IDF + MLP
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training Model Family B  (TF-IDF + MLP, hidden=32, relu, adam) ...")
    print("-" * 70)

    model_b = ModelFamilyB(categories=CATEGORIES)
    model_b.fit(texts, labels_dict)

    print("\nTrain-split 5-fold CV F1 (Family B):")
    for cat in CATEGORIES:
        y = np.array(labels_dict[cat])
        if sum(y) < 5:
            print(f"  {cat:<25}: SKIPPED (fewer than 5 positives)")
            continue
        pipe = model_b.pipelines[cat]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, texts, y, cv=cv, scoring="f1", error_score=0.0)
        print(f"  {cat:<25}: F1 = {np.mean(scores):.3f}  (+/-{np.std(scores):.3f})")

    # -------------------------------------------------------------------------
    # Save models
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"Saving models to '{MODELS_DIR}/' ...")
    model_a.save(MODELS_DIR)
    model_b.save(MODELS_DIR)

    print("\n" + "=" * 70)
    print("Phase 2 training complete.")
    print("Models now trained on data/train.json ONLY (645 samples, seed=42).")
    print("Next: run phase2_benchmark.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
