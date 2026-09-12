"""
ToxicBert Classifier: Wraps unitary/toxic-bert for production-grade hate speech and toxicity detection.
Model outputs 6 labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Maps to CipherGuard 'Hate/Toxicity' via: max(toxic, severe_toxic, threat, identity_hate).
Scores only 'Hate/Toxicity'; returns 0.0 for all other categories.
Provides keyword-heuristic fallback if transformers/torch not available.
"""

from typing import Dict, List, Optional
import torch
from .base import BaseClassifier

MODEL_NAME = "unitary/toxic-bert"

ALL_CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
CATEGORY = "Hate/Toxicity"

# Indices of output labels from unitary/toxic-bert that map to our hate/toxicity signal
# Label order: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
HATE_LABEL_INDICES = [0, 1, 3, 5]  # toxic, severe_toxic, threat, identity_hate

# Fallback keyword triggers
FALLBACK_TRIGGERS = [
    "kill all", "hate speech", "racial slur", "terrorist attack", "exterminate",
    "genocide", "slaughter", "death threat", "i will kill", "bomb threat",
    "ethnic cleansing", "white supremac", "neo nazi", "go kill yourself",
    "you deserve to die", "i hate all", "shoot them all"
]


class ToxicBertClassifier(BaseClassifier):
    """
    Hate speech and toxicity detector using unitary/toxic-bert.

    Replaces the keyword-trigger heuristic in ModelFamilyA/B for the Hate/Toxicity category.
    Uses sigmoid outputs (multi-label) and maps relevant labels to a single risk score.

    Score formula: score = max(toxic, severe_toxic, threat, identity_hate) probabilities
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None, lazy_load: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._fallback_mode = False
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._is_loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[CipherGuard] Loading toxic-bert: {self.model_name} on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
            self._fallback_mode = False
            print("[CipherGuard] ToxicBert loaded successfully.")
        except ImportError:
            print("[CipherGuard] transformers not installed. Hate/Toxicity using keyword fallback.")
            self._fallback_mode = True
            self._is_loaded = True
        except Exception as e:
            print(f"[CipherGuard] ToxicBert failed to load ({e}). Using keyword fallback.")
            self._fallback_mode = True
            self._is_loaded = True

    def _keyword_fallback(self, text: str) -> float:
        """Keyword-based toxicity heuristic."""
        t = text.lower()
        hits = sum(1 for trigger in FALLBACK_TRIGGERS if trigger in t)
        if hits > 0:
            return min(1.0, 0.75 + 0.05 * (hits - 1))
        return 0.02

    def _infer_batch(self, texts: List[str]) -> List[float]:
        """Run batched toxic-bert inference and return Hate/Toxicity scores."""
        results = []
        batch_size = 16  # Smaller batch for memory efficiency
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            try:
                inputs = self._tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # unitary/toxic-bert uses sigmoid (multi-label), not softmax
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()

                for prob_row in probs:
                    # prob_row shape: [6] = [toxic, severe_toxic, obscene, threat, insult, identity_hate]
                    hate_score = float(max(prob_row[idx] for idx in HATE_LABEL_INDICES))
                    results.append(round(hate_score, 4))
            except Exception as e:
                print(f"[CipherGuard] ToxicBert inference error: {e}")
                for t in chunk:
                    results.append(self._keyword_fallback(t))
        return results

    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """Score a single text. Only 'Hate/Toxicity' is non-zero."""
        if not self._is_loaded:
            self._load_model()

        base_scores = {cat: 0.0 for cat in ALL_CATEGORIES}

        if not text or not text.strip():
            return base_scores

        if self._fallback_mode or self._model is None:
            hate_score = self._keyword_fallback(text)
        else:
            scores = self._infer_batch([text])
            hate_score = scores[0] if scores else self._keyword_fallback(text)

        base_scores[CATEGORY] = hate_score
        return base_scores

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """Vectorized batch hate/toxicity scoring."""
        if not self._is_loaded:
            self._load_model()

        if not texts:
            return []

        base_results = [{cat: 0.0 for cat in ALL_CATEGORIES} for _ in texts]

        if self._fallback_mode or self._model is None:
            for i, text in enumerate(texts):
                base_results[i][CATEGORY] = self._keyword_fallback(text)
        else:
            hate_scores = self._infer_batch(texts)
            for i, score in enumerate(hate_scores):
                base_results[i][CATEGORY] = score

        return base_results
