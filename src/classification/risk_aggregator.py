"""
Risk Aggregator: Ensembles Family A and Family B models with conservative calibration.
Implements s(category) = max(score_A, score_B) while logging raw scores for auditing.
Matches Protocol §6.1.
"""

from typing import Dict, List, Optional, Any
from .base import BaseClassifier
from .model_family_a import ModelFamilyA
from .model_family_b import ModelFamilyB


class RiskAggregator:
    """
    Coordinates multi-model classification and score fusion.
    """

    def __init__(self, model_a: Optional[BaseClassifier] = None, model_b: Optional[BaseClassifier] = None):
        self.model_a = model_a or ModelFamilyA()
        self.model_b = model_b or ModelFamilyB()

    def aggregate(self, text: str, surface: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs both models, records individual scores, and computes conservative aggregate:
        s(category) = max(score_a, score_b).
        """
        score_a = self.model_a.score(text, surface)
        score_b = self.model_b.score(text, surface)

        all_categories = sorted(list(set(score_a.keys()) | set(score_b.keys())))
        fused_scores: Dict[str, float] = {}

        for cat in all_categories:
            val_a = score_a.get(cat, 0.0)
            val_b = score_b.get(cat, 0.0)
            fused_scores[cat] = round(max(val_a, val_b), 4)

        return {
            "fused_scores": fused_scores,
            "raw_scores": {
                "model_a": score_a,
                "model_b": score_b
            },
            "surface": surface
        }

    def aggregate_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Vectorized batch aggregation across both model families.
        """
        scores_a = self.model_a.score_batch(texts, surfaces)
        scores_b = self.model_b.score_batch(texts, surfaces)

        results = []
        for i in range(len(texts)):
            s_a = scores_a[i]
            s_b = scores_b[i]
            surf = surfaces[i] if surfaces else None
            all_cats = sorted(list(set(s_a.keys()) | set(s_b.keys())))
            fused = {cat: round(max(s_a.get(cat, 0.0), s_b.get(cat, 0.0)), 4) for cat in all_cats}
            results.append({
                "fused_scores": fused,
                "raw_scores": {
                    "model_a": s_a,
                    "model_b": s_b
                },
                "surface": surf
            })
        return results
