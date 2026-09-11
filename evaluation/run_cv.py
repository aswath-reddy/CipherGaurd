"""
Repeated Cross-Validation Harness for Model Family A (LogReg) and Family B (MLP).
Replicates Section IV-C and Table I from the research paper.
"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .build_dataset import load_pilot_dataset


def evaluate_repeated_cv(n_splits: int = 5, n_repeats: int = 5):
    """
    Executes 5x5 repeated cross-validation across 25 folds.
    """
    data = load_pilot_dataset()
    texts = [d["text"] for d in data]
    labels = np.array([d["label"] for d in data])
    surfaces = np.array([d["surface"] for d in data])

    print(f"=== CipherGuard Cross-Validation Benchmark (n={len(data)}) ===")
    print(f"Folds: {n_splits}, Repeats: {n_repeats} (Total folds: {n_splits * n_repeats})")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    metrics_logreg = {"acc": [], "prec": [], "rec": [], "f1": []}
    metrics_mlp = {"acc": [], "prec": [], "rec": [], "f1": []}

    fold_idx = 0
    for train_idx, test_idx in rskf.split(texts, labels):
        fold_idx += 1
        X_train = [texts[i] for i in train_idx]
        y_train = labels[train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = labels[test_idx]

        # Model Family A: Logistic Regression
        pipe_lr = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(C=1.0, max_iter=200, class_weight="balanced"))
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)

        metrics_logreg["acc"].append(accuracy_score(y_test, y_pred_lr))
        metrics_logreg["prec"].append(precision_score(y_test, y_pred_lr, zero_division=0))
        metrics_logreg["rec"].append(recall_score(y_test, y_pred_lr, zero_division=0))
        metrics_logreg["f1"].append(f1_score(y_test, y_pred_lr, zero_division=0))

        # Model Family B: Multilayer Perceptron (MLP)
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

    def summarize(m: dict) -> dict:
        return {
            "Accuracy": f"{np.mean(m['acc']):.3f} (±{np.std(m['acc']):.3f})",
            "Precision": f"{np.mean(m['prec']):.3f} (±{np.std(m['prec']):.3f})",
            "Recall": f"{np.mean(m['rec']):.3f} (±{np.std(m['rec']):.3f})",
            "F1": f"{np.mean(m['f1']):.3f} (±{np.std(m['f1']):.3f})",
        }

    res_lr = summarize(metrics_logreg)
    res_mlp = summarize(metrics_mlp)

    print("\n--- TABLE I REPRODUCTION RESULTS ---")
    print(f"{'Metric':<15} | {'LogReg (Family A)':<22} | {'MLP (Family B)':<22}")
    print("-" * 65)
    for k in ["Accuracy", "Precision", "Recall", "F1"]:
        print(f"{k:<15} | {res_lr[k]:<22} | {res_mlp[k]:<22}")

    return {"LogReg": res_lr, "MLP": res_mlp}


if __name__ == "__main__":
    evaluate_repeated_cv()
