"""
Zero-Shot Comparison Runner for DeBERTa Semantic Classifier.
Evaluates protectai/deberta-v3-base-prompt-injection-v2 on n=85 dataset.
Replicates Section IV-G and Table II from the research paper.
"""

from typing import List, Dict, Any
from .build_dataset import load_pilot_dataset
from .metrics import compute_classification_metrics, compute_attribution_metrics
from src.classification.deberta_model import DebertaSemanticClassifier
from src.attribution import run_removal_beam_search


def run_zero_shot_comparison():
    data = load_pilot_dataset()
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    surfaces = [d["surface"] for d in data]

    print(f"=== CipherGuard Zero-Shot Semantic Classifier Benchmark (n={len(data)}) ===")
    classifier = DebertaSemanticClassifier()

    # Step 1: Batched scoring across all examples
    print("Scoring dataset with DeBERTa-v3 in batched mode...")
    scored_results = classifier.score_batch(texts, surfaces)
    preds_50 = []
    direct_true, direct_pred = [], []
    indirect_true, indirect_pred = [], []

    scores_list = []
    for d, s in zip(data, scored_results):
        score = s.get("Prompt Injection", 0.0)
        scores_list.append(score)
        pred = 1 if score >= 0.50 else 0
        preds_50.append(pred)

        if d["surface"] == "direct":
            direct_true.append(d["label"])
            direct_pred.append(pred)
        else:
            indirect_true.append(d["label"])
            indirect_pred.append(pred)

    overall_metrics = compute_classification_metrics(labels, preds_50)
    direct_metrics = compute_classification_metrics(direct_true, direct_pred)
    indirect_metrics = compute_classification_metrics(indirect_true, indirect_pred)

    print("\n--- TABLE II REPRODUCTION: CLASSIFICATION ---")
    print(f"Overall Accuracy:  {overall_metrics['accuracy']:.3f}")
    print(f"Overall Precision: {overall_metrics['precision']:.3f}")
    print(f"Overall Recall:    {overall_metrics['recall']:.3f}")
    print(f"Direct Recall:     {direct_metrics['recall']:.3f}")
    print(f"Indirect Recall:   {indirect_metrics['recall']:.3f}")
    print(f"(Surface Asymmetry: Direct={direct_metrics['recall']:.3f} vs Indirect={indirect_metrics['recall']:.3f})")

    # Step 2: Attribution search on blocked true positives
    blocked_indices = [i for i, (pred, y) in enumerate(zip(preds_50, labels)) if pred == 1 and y == 1]
    print(f"\nRunning batched removal attribution on {len(blocked_indices)} blocked instances...")

    def deberta_batch_score(cands: List[str]) -> List[float]:
        res = classifier.score_batch(cands)
        return [r.get("Prompt Injection", 0.0) for r in res]

    attribution_results = []
    for idx in blocked_indices:
        text = texts[idx]
        res = run_removal_beam_search(
            text=text,
            score_fn=deberta_batch_score,
            threshold=0.45,
            beam_width=5,
            removal_cap=6
        )
        attribution_results.append(res.to_dict())

    att_metrics = compute_attribution_metrics(attribution_results)

    print("\n--- TABLE II REPRODUCTION: ATTRIBUTION ---")
    print(f"Flip Rate (beam=5, cap<=6): {att_metrics['flip_rate']*100:.1f}% ({att_metrics['successful_flips']}/{att_metrics['total_evaluated']})")
    print(f"Mean Tokens Removed:       {att_metrics['mean_tokens_removed']:.2f}")
    print(f"Mean Attribution Latency:  {att_metrics['mean_latency_ms']:.2f} ms")

    return {
        "classification": overall_metrics,
        "direct_recall": direct_metrics["recall"],
        "indirect_recall": indirect_metrics["recall"],
        "attribution": att_metrics
    }


if __name__ == "__main__":
    run_zero_shot_comparison()
