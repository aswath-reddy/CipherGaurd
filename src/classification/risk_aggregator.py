"""
Risk Aggregator: Ensembles all model families with per-category specialist routing.

Fusion strategy per category:
  Jailbreak / Prompt Injection : max(Family_A, Family_B, DeBERTa)
  PII Leakage                  : max(Family_A, Family_B, Presidio)
  Malicious Tools              : max(Family_A, Family_B)
  Hate/Toxicity                : max(Family_A, Family_B, ToxicBert)

Matches Protocol §6.1 (extended with specialist classifiers).
"""

from typing import Dict, List, Optional, Any
from .base import BaseClassifier
from .model_family_a import ModelFamilyA
from .model_family_b import ModelFamilyB
from .finetuned_deberta_classifier import FinetunedDebertaClassifier
from .presidio_classifier import PresidioClassifier
from .toxic_bert_classifier import ToxicBertClassifier

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]

# Map category -> which specialist classifiers contribute (in addition to Family A & B)
SPECIALIST_CATEGORIES = {
    "Jailbreak":        ["deberta"],
    "Prompt Injection": ["deberta"],
    "PII Leakage":      ["presidio"],
    "Malicious Tools":  [],
    "Hate/Toxicity":    ["toxic_bert"],
}


class RiskAggregator:
    """
    Coordinates multi-model classification with per-category specialist routing.

    Each risk category is scored by Family A and Family B (general discriminators)
    PLUS a dedicated specialist model tuned for that threat type:
      - DeBERTa-v3  for Jailbreak and Prompt Injection
      - Presidio     for PII Leakage
      - ToxicBert   for Hate/Toxicity

    Final score per category: max over all contributing classifiers.
    All specialist classifiers lazy-load on first use.
    """

    def __init__(
        self,
        model_a: Optional[BaseClassifier] = None,
        model_b: Optional[BaseClassifier] = None,
        deberta: Optional[BaseClassifier] = None,
        presidio: Optional[BaseClassifier] = None,
        toxic_bert: Optional[BaseClassifier] = None,
        models_dir: str = "models"
    ):
        import os
        if model_a is not None:
            self.model_a = model_a
        elif os.path.exists(models_dir) and any(f.startswith("family_a_") for f in os.listdir(models_dir)):
            self.model_a = ModelFamilyA.load(models_dir)
        else:
            self.model_a = ModelFamilyA()

        if model_b is not None:
            self.model_b = model_b
        elif os.path.exists(models_dir) and any(f.startswith("family_b_") for f in os.listdir(models_dir)):
            self.model_b = ModelFamilyB.load(models_dir)
        else:
            self.model_b = ModelFamilyB()

        self.deberta = deberta or FinetunedDebertaClassifier()   # lazy-loads fine-tuned or base
        self.presidio = presidio or PresidioClassifier()          # lazy-loads Presidio
        self.toxic_bert = toxic_bert or ToxicBertClassifier()     # lazy-loads ToxicBert

        # Specialist registry for routing
        self._specialists = {
            "deberta":   self.deberta,
            "presidio":  self.presidio,
            "toxic_bert": self.toxic_bert,
        }

    def _fuse(self, score_a: Dict, score_b: Dict, specialist_scores: Dict[str, Dict]) -> Dict[str, float]:
        """
        Fuse scores per category using max-pooling across all contributing classifiers.
        """
        fused: Dict[str, float] = {}
        for cat in CATEGORIES:
            candidates = [
                score_a.get(cat, 0.0),
                score_b.get(cat, 0.0),
            ]
            # Add specialist signals for this category
            for specialist_key in SPECIALIST_CATEGORIES.get(cat, []):
                spec_scores = specialist_scores.get(specialist_key, {})
                candidates.append(spec_scores.get(cat, 0.0))

            fused[cat] = round(max(candidates), 4)
        return fused

    def aggregate(self, text: str, surface: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs all models and computes per-category max-pooled risk scores.
        """
        score_a = self.model_a.score(text, surface)
        score_b = self.model_b.score(text, surface)

        # Run specialist classifiers (only load what's needed)
        specialist_scores: Dict[str, Dict] = {}
        for spec_key, specialist in self._specialists.items():
            try:
                specialist_scores[spec_key] = specialist.score(text, surface)
            except Exception as e:
                print(f"[CipherGuard] Specialist '{spec_key}' error: {e}")
                specialist_scores[spec_key] = {cat: 0.0 for cat in CATEGORIES}

        fused = self._fuse(score_a, score_b, specialist_scores)

        return {
            "fused_scores": fused,
            "raw_scores": {
                "model_a": score_a,
                "model_b": score_b,
                "deberta": specialist_scores.get("deberta", {}),
                "presidio": specialist_scores.get("presidio", {}),
                "toxic_bert": specialist_scores.get("toxic_bert", {}),
            },
            "surface": surface
        }

    def aggregate_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Vectorized batch aggregation across all model families and specialists.
        """
        if not texts:
            return []

        scores_a = self.model_a.score_batch(texts, surfaces)
        scores_b = self.model_b.score_batch(texts, surfaces)

        # Batch specialist scoring
        batch_specialist: Dict[str, List[Dict]] = {}
        for spec_key, specialist in self._specialists.items():
            try:
                batch_specialist[spec_key] = specialist.score_batch(texts, surfaces)
            except Exception as e:
                print(f"[CipherGuard] Batch specialist '{spec_key}' error: {e}")
                batch_specialist[spec_key] = [{cat: 0.0 for cat in CATEGORIES} for _ in texts]

        results = []
        for i in range(len(texts)):
            s_a = scores_a[i]
            s_b = scores_b[i]
            surf = surfaces[i] if surfaces else None
            specialist_scores_i = {k: v[i] for k, v in batch_specialist.items()}

            fused = self._fuse(s_a, s_b, specialist_scores_i)
            results.append({
                "fused_scores": fused,
                "raw_scores": {
                    "model_a": s_a,
                    "model_b": s_b,
                    **{k: specialist_scores_i[k] for k in self._specialists}
                },
                "surface": surf
            })

        return results
