"""
Evaluation Metrics Calculator for Classification and Contrastive Attribution.
Matches Paper Tables I & II and Section IV specifications.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes standard classification metrics: Accuracy, Precision, Recall, F1.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    }


def compute_attribution_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes attribution evaluation metrics:
    - Flip Rate (% flipped to PASS within budget)
    - Mean and Median Cardinality (|ΔS| tokens removed)
    - Mean and P95 Search Latency (ms)
    """
    if not results:
        return {}

    total = len(results)
    successful_flips = [r for r in results if r.get("flipped")]
    flip_rate = len(successful_flips) / total

    cardinalities = [r.get("cardinality", 0) for r in successful_flips]
    latencies = [r.get("search_latency_ms", 0.0) for r in results]

    mean_cardinality = float(np.mean(cardinalities)) if cardinalities else 0.0
    median_cardinality = float(np.median(cardinalities)) if cardinalities else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "total_evaluated": total,
        "successful_flips": len(successful_flips),
        "flip_rate": round(flip_rate, 4),
        "mean_tokens_removed": round(mean_cardinality, 2),
        "median_tokens_removed": round(median_cardinality, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2)
    }
