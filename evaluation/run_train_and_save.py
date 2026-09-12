"""
Production Training Script for CipherGuard Model Family A and Family B.

Trains one TF-IDF pipeline per category for both Family A (LogReg) and Family B (MLP)
using the expanded multi-label dataset, then saves fitted pipelines to disk.

Usage:
    python -m evaluation.run_train_and_save

Output:
    models/family_a_{category}.pkl
    models/family_b_{category}.pkl
"""

import os
import sys
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Allow running from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from evaluation.build_dataset import load_expanded_dataset, get_category_labels
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
MODELS_DIR = "models"


def train_and_evaluate():
    print("=" * 65)
    print("CipherGuard - Training Family A (LogReg) & Family B (MLP)")
    print("=" * 65)

    # Load dataset
    data = load_expanded_dataset()
    if not data:
        print("ERROR: No data loaded. Run build_expanded_dataset.py first.")
        return

    texts = [d["text"] for d in data]
    print(f"\nDataset loaded: {len(data)} samples")

    # Print class distribution
    print("\nClass distribution per category:")
    for cat in CATEGORIES:
        y = get_category_labels(data, cat)
        pos = sum(y)
        print(f"  {cat:<25}: {pos:>4} positive / {len(y)-pos:>4} negative")

    # Initialize models
    model_a = ModelFamilyA(categories=CATEGORIES)
    model_b = ModelFamilyB(categories=CATEGORIES)

    # Build labels dict
    labels_dict = {cat: get_category_labels(data, cat) for cat in CATEGORIES}

    # ─────────────────────────────────────
    # Train Family A
    # ─────────────────────────────────────
    print("\n" + "-" * 65)
    print("Training Model Family A (TF-IDF + Logistic Regression)...")
    print("-" * 65)
    model_a.fit(texts, labels_dict)

    print("\nPer-category cross-validation (3-fold, F1):")
    for cat in CATEGORIES:
        y = np.array(labels_dict[cat])
        if sum(y) < 3:
            print(f"  {cat:<25}: (skipped - insufficient positives)")
            continue
        pipe = model_a.pipelines[cat]
        cv_scores = cross_val_score(pipe, texts, y, cv=3, scoring="f1", error_score=0.0)
        print(f"  {cat:<25}: F1 = {np.mean(cv_scores):.3f} (+/-{np.std(cv_scores):.3f})")

    # ─────────────────────────────────────
    # Train Family B
    # ─────────────────────────────────────
    print("\n" + "-" * 65)
    print("Training Model Family B (TF-IDF + MLP)...")
    print("-" * 65)
    model_b.fit(texts, labels_dict)

    print("\nPer-category cross-validation (3-fold, F1):")
    for cat in CATEGORIES:
        y = np.array(labels_dict[cat])
        if sum(y) < 3:
            print(f"  {cat:<25}: (skipped - insufficient positives)")
            continue
        pipe = model_b.pipelines[cat]
        cv_scores = cross_val_score(pipe, texts, y, cv=3, scoring="f1", error_score=0.0)
        print(f"  {cat:<25}: F1 = {np.mean(cv_scores):.3f} (+/-{np.std(cv_scores):.3f})")

    # ─────────────────────────────────────
    # Save models
    # ─────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"Saving trained models to '{MODELS_DIR}/'...")
    model_a.save(MODELS_DIR)
    model_b.save(MODELS_DIR)

    print("\n" + "=" * 65)
    print("Training complete. Models saved.")
    print("\nTo use trained models in production, load with:")
    print("  from src.classification import ModelFamilyA, ModelFamilyB")
    print("  model_a = ModelFamilyA.load('models/')")
    print("  model_b = ModelFamilyB.load('models/')")
    print("  aggregator = RiskAggregator(model_a=model_a, model_b=model_b)")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate()
