"""
Attribution Search Budget Ablation Runner.
Compares Narrow Budget (beam=5, cap=6) vs Wide Budget (beam=10, cap=10).
Replicates Section IV-D and Table I from the research paper.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .build_dataset import load_pilot_dataset
from .metrics import compute_attribution_metrics
from src.attribution import run_removal_beam_search


def run_budget_ablation():
    data = load_pilot_dataset()
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    # Fit a standard baseline classifier to serve as the score function
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(C=1.0, max_iter=200, class_weight="balanced"))
    ])
    pipe.fit(texts, labels)

    def score_batch(cands: List[str]) -> List[float]:
        probs = pipe.predict_proba(cands)[:, 1]
        return [float(p) for p in probs]

    # Filter blocked examples (predicted unsafe @ 0.50 threshold)
    blocked_examples = []
    for d in data:
        score = score_batch([d["text"]])[0]
        if score >= 0.50:
            blocked_examples.append(d["text"])

    print(f"=== CipherGuard Attribution Budget Ablation ===")
    print(f"Total Blocked Examples Evaluated: {len(blocked_examples)}")

    threshold = 0.45

    # 1. Narrow Budget: beam=5, cap=6
    print("\n[1/2] Evaluating Narrow Budget (beam_width=5, removal_cap=6)...")
    results_narrow: List[Dict[str, Any]] = []
    for text in blocked_examples:
        res = run_removal_beam_search(
            text=text,
            score_fn=score_batch,
            threshold=threshold,
            beam_width=5,
            removal_cap=6
        )
        results_narrow.append(res.to_dict())

    metrics_narrow = compute_attribution_metrics(results_narrow)

    # 2. Wide Budget: beam=10, cap=10
    print("[2/2] Evaluating Wide Budget (beam_width=10, removal_cap=10)...")
    results_wide: List[Dict[str, Any]] = []
    for text in blocked_examples:
        res = run_removal_beam_search(
            text=text,
            score_fn=score_batch,
            threshold=threshold,
            beam_width=10,
            removal_cap=10
        )
        results_wide.append(res.to_dict())

    metrics_wide = compute_attribution_metrics(results_wide)

    print("\n--- BUDGET ABLATION COMPARISON ---")
    print(f"{'Budget Configuration':<28} | {'Flip Rate':<12} | {'Mean Tokens Removed':<22} | {'Mean Latency':<15}")
    print("-" * 85)
    print(
        f"{'Narrow (beam=5, cap=6)':<28} | "
        f"{metrics_narrow['flip_rate']*100:.1f}% ({metrics_narrow['successful_flips']}/{metrics_narrow['total_evaluated']}){' ':<2} | "
        f"{metrics_narrow['mean_tokens_removed']:<22} | "
        f"{metrics_narrow['mean_latency_ms']:.1f} ms"
    )
    print(
        f"{'Wide (beam=10, cap=10)':<28} | "
        f"{metrics_wide['flip_rate']*100:.1f}% ({metrics_wide['successful_flips']}/{metrics_wide['total_evaluated']}){' ':<2} | "
        f"{metrics_wide['mean_tokens_removed']:<22} | "
        f"{metrics_wide['mean_latency_ms']:.1f} ms"
    )

    return {"narrow": metrics_narrow, "wide": metrics_wide}


if __name__ == "__main__":
    run_budget_ablation()
