"""
Repeated Cross-Validation Harness for Model Family A (LogReg) and Family B (MLP).
Supports both legacy binary-label and new multi-label expanded dataset.
Replicates and extends Section IV-C from the research paper.
"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .build_dataset import load_expanded_dataset, load_pilot_dataset, get_category_labels

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]


def evaluate_category_cv(texts, labels, category_name, n_splits=5, n_repeats=3):
    """
    Runs repeated cross-validation for a single category.
    Returns mean and std of Accuracy, Precision, Recall, F1.
    """
    y = np.array(labels)
    if sum(y) < n_splits:
        return None  # Not enough positive samples

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    metrics_lr = {"acc": [], "prec": [], "rec": [], "f1": []}
    metrics_mlp = {"acc": [], "prec": [], "rec": [], "f1": []}

    for train_idx, test_idx in rskf.split(texts, y):
        X_train = [texts[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = y[test_idx]

        # Family A: Logistic Regression
        pipe_lr = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(C=1.0, max_iter=200, class_weight="balanced"))
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        metrics_lr["acc"].append(accuracy_score(y_test, y_pred_lr))
        metrics_lr["prec"].append(precision_score(y_test, y_pred_lr, zero_division=0))
        metrics_lr["rec"].append(recall_score(y_test, y_pred_lr, zero_division=0))
        metrics_lr["f1"].append(f1_score(y_test, y_pred_lr, zero_division=0))

        # Family B: MLP
        pipe_mlp = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", MLPClassifier(hidden_layer_sizes=(32,), activation="relu", max_iter=500, random_state=42))
        ])
        pipe_mlp.fit(X_train, y_train)
        y_pred_mlp = pipe_mlp.predict(X_test)
        metrics_mlp["acc"].append(accuracy_score(y_test, y_pred_mlp))
        metrics_mlp["prec"].append(precision_score(y_test, y_pred_mlp, zero_division=0))
        metrics_mlp["rec"].append(recall_score(y_test, y_pred_mlp, zero_division=0))
        metrics_mlp["f1"].append(f1_score(y_test, y_pred_mlp, zero_division=0))

    def fmt(m):
        return {
            "Accuracy":  f"{np.mean(m['acc']):.3f} (+/-{np.std(m['acc']):.3f})",
            "Precision": f"{np.mean(m['prec']):.3f} (+/-{np.std(m['prec']):.3f})",
            "Recall":    f"{np.mean(m['rec']):.3f} (+/-{np.std(m['rec']):.3f})",
            "F1":        f"{np.mean(m['f1']):.3f} (+/-{np.std(m['f1']):.3f})",
        }

    return {"LogReg": fmt(metrics_lr), "MLP": fmt(metrics_mlp)}


def evaluate_repeated_cv(n_splits: int = 5, n_repeats: int = 3, use_expanded: bool = True):
    """
    Runs per-category repeated cross-validation on the expanded dataset.
    """
    data = load_expanded_dataset() if use_expanded else load_pilot_dataset()
    texts = [d["text"] for d in data]

    dataset_label = "expanded" if use_expanded else "pilot (n=85)"
    print(f"=== CipherGuard Per-Category Cross-Validation (dataset={dataset_label}, n={len(data)}) ===")
    print(f"Folds: {n_splits}, Repeats: {n_repeats} (Total folds: {n_splits * n_repeats})")

    all_results = {}
    for cat in CATEGORIES:
        labels = get_category_labels(data, cat)
        pos = sum(labels)
        print(f"\n--- Category: {cat} ({pos} positive / {len(labels)-pos} negative) ---")
        result = evaluate_category_cv(texts, labels, cat, n_splits=n_splits, n_repeats=n_repeats)
        if result is None:
            print(f"  SKIPPED: fewer than {n_splits} positive samples available.")
            continue

        all_results[cat] = result
        print(f"  {'Metric':<12} | {'LogReg (Family A)':<24} | {'MLP (Family B)':<24}")
        print("  " + "-" * 66)
        for metric in ["Accuracy", "Precision", "Recall", "F1"]:
            lr_val = result["LogReg"][metric]
            mlp_val = result["MLP"][metric]
            print(f"  {metric:<12} | {lr_val:<24} | {mlp_val:<24}")

    return all_results


if __name__ == "__main__":
    evaluate_repeated_cv()
