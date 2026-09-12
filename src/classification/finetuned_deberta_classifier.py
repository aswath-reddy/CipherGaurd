"""
Fine-tuned DeBERTa Multi-Label Classifier.
Loads a fine-tuned model from models/cipherguard-deberta-finetuned/ when available.
Falls back to the zero-shot base DeBERTa if the fine-tuned model is not present.
Outputs proper per-category sigmoid probabilities for all 5 risk categories.
"""

import os
from typing import Dict, List, Optional
import torch
from .base import BaseClassifier

FINETUNED_MODEL_DIR = "models/cipherguard-deberta-finetuned"
BASE_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]


class FinetunedDebertaClassifier(BaseClassifier):
    """
    Multi-label DeBERTa classifier with 5-category output.

    When models/cipherguard-deberta-finetuned/ exists (after running finetune_deberta.py
    on a GPU system), loads the fine-tuned model and uses sigmoid outputs per category.

    When the fine-tuned model is not present, automatically falls back to the base
    DeBERTa (protectai/deberta-v3-base-prompt-injection-v2) in zero-shot mode,
    which only provides meaningful Jailbreak/Injection scores.
    """

    def __init__(
        self,
        model_dir: str = FINETUNED_MODEL_DIR,
        base_model_name: str = BASE_MODEL_NAME,
        device: Optional[str] = None,
        lazy_load: bool = True
    ):
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._is_finetuned = False
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._is_loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Check if fine-tuned model exists
            if os.path.isdir(self.model_dir) and os.path.exists(
                os.path.join(self.model_dir, "config.json")
            ):
                print(f"[CipherGuard] Loading fine-tuned DeBERTa from {self.model_dir}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self._is_finetuned = True
                print("[CipherGuard] Fine-tuned DeBERTa (5-category multi-label) loaded.")
            else:
                print(f"[CipherGuard] Fine-tuned model not found at '{self.model_dir}'.")
                print(f"[CipherGuard] Loading base model: {self.base_model_name} (zero-shot injection only).")
                print(f"[CipherGuard] To get full 5-category support, run: python -m evaluation.finetune_deberta")
                self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)
                self._is_finetuned = False

            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
        except Exception as e:
            print(f"[CipherGuard] FinetunedDebertaClassifier failed to load: {e}")
            self._is_loaded = False

    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        batch_res = self.score_batch([text], [surface] if surface else None)
        return batch_res[0] if batch_res else {cat: 0.0 for cat in CATEGORIES}

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        if not texts:
            return []
        if not self._is_loaded:
            self._load_model()
        if not self._is_loaded or self._model is None:
            return [{cat: 0.0 for cat in CATEGORIES} for _ in texts]

        results = []
        batch_size = 16
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
                    logits = outputs.logits

                if self._is_finetuned:
                    # Fine-tuned model: sigmoid over 5-category logits
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for prob_row in probs:
                        scores = {cat: round(float(p), 4) for cat, p in zip(CATEGORIES, prob_row)}
                        results.append(scores)
                else:
                    # Base model: binary injection, map to injection/jailbreak only
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    for p in probs:
                        prob_val = round(float(p), 4)
                        results.append({
                            "Jailbreak": prob_val,
                            "Prompt Injection": prob_val,
                            "PII Leakage": 0.01,
                            "Malicious Tools": 0.02,
                            "Hate/Toxicity": 0.01
                        })
            except Exception as e:
                print(f"[CipherGuard] FinetunedDeberta inference error: {e}")
                for _ in chunk:
                    results.append({cat: 0.0 for cat in CATEGORIES})

        return results
