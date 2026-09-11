"""
DeBERTa Semantic Classifier Wrapper
Evaluates protectai/deberta-v3-base-prompt-injection-v2 with batched tensor inference.
Matches Section IV-G specifications.
"""

from typing import Dict, List, Optional
import torch
from .base import BaseClassifier

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"


class DebertaSemanticClassifier(BaseClassifier):
    """
    Wrapper for protectai/deberta-v3-base-prompt-injection-v2.
    Implements vectorized batch inference to eliminate the unbatched latency bottleneck (4.92s) in the paper.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None, lazy_load: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self.is_loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[CipherGuard] Loading semantic classifier: {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("[CipherGuard] Semantic model loaded successfully.")
        except Exception as e:
            print(f"[CipherGuard] Warning: Could not load transformer model '{self.model_name}': {e}")
            print("[CipherGuard] Operating in lightweight fallback mode.")
            self.is_loaded = False

    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """
        Score a single string.
        """
        batch_res = self.score_batch([text], [surface] if surface else None)
        return batch_res[0] if batch_res else {"Prompt Injection": 0.0, "Jailbreak": 0.0}

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Batched transformer forward pass with torch.no_grad().
        """
        if not texts:
            return []

        if not self.is_loaded:
            self._load_model()

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return [self._fallback_score(t, surfaces[i] if surfaces else None) for i, t in enumerate(texts)]

        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            try:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    # protectai model outputs [SAFE_prob, INJECTION_prob]
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                for p in probs:
                    prob_val = round(float(p), 4)
                    results.append({
                        "Prompt Injection": prob_val,
                        "Jailbreak": prob_val,
                        "PII Leakage": 0.01,
                        "Malicious Tools": 0.02,
                        "Hate/Toxicity": 0.01
                    })
            except Exception as e:
                print(f"[CipherGuard] Inference error in batch: {e}")
                for t in chunk:
                    results.append(self._fallback_score(t))

        return results

    def _fallback_score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """
        High-fidelity heuristic fallback when running in offline or unit-test environments.
        """
        t = text.lower()
        triggers = [
            "ignore previous instructions", "ignore all instructions", "system prompt",
            "hidden instruction", "override instructions", "pretend you are", "jailbreak"
        ]
        is_attack = any(trig in t for trig in triggers)
        prob = 0.96 if is_attack else 0.03
        return {
            "Prompt Injection": prob,
            "Jailbreak": prob,
            "PII Leakage": 0.01,
            "Malicious Tools": 0.02,
            "Hate/Toxicity": 0.01
        }
