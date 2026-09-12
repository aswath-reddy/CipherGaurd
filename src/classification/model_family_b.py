"""
Model Family B: Nonlinear Multilayer Perceptron Discriminator (TF-IDF + MLP / MiniLM interface)
Matches Section IV-A and Protocol Family B specifications.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from .base import BaseClassifier


class ModelFamilyB(BaseClassifier):
    """
    Family B classifier: Small Multilayer Perceptron (MLP).
    Uses unigram & bigram TF-IDF with a 32-unit hidden layer, ReLU activation, and Adam optimizer.
    Represents a nonlinear model family trading off higher recall for different operating bounds.
    """

    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
        self.pipelines: Dict[str, Pipeline] = {}
        self.is_fitted = False
        self._init_default_pipelines()

    def _init_default_pipelines(self):
        for cat in self.categories:
            self.pipelines[cat] = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)),
                ("mlp", MLPClassifier(hidden_layer_sizes=(32,), activation="relu", max_iter=200, random_state=42))
            ])

    def fit(self, texts: List[str], labels: Dict[str, List[int]]):
        """
        Fit an MLP pipeline for each risk category.
        """
        for cat, y in labels.items():
            if cat in self.pipelines:
                if len(set(y)) >= 2:
                    self.pipelines[cat].fit(texts, y)
                else:
                    print(f"[CipherGuard] ModelFamilyB: Category '{cat}' has only 1 class in training data. Keeping fallback heuristic.")
        self.is_fitted = True

    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """
        Returns risk scores in [0, 1] per category.
        """
        scores: Dict[str, float] = {}
        clean_text = text.strip()
        if not clean_text:
            return {cat: 0.0 for cat in self.categories}

        for cat in self.categories:
            pipe = self.pipelines.get(cat)
            if pipe is not None and self.is_fitted:
                try:
                    prob = float(pipe.predict_proba([clean_text])[0][1])
                except Exception:
                    prob = 0.0
            else:
                prob = self._heuristic_fallback(clean_text, cat)
            scores[cat] = round(prob, 4)
        return scores

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Vectorized batch scoring for MLP.
        """
        if not texts:
            return []
        
        if self.is_fitted:
            results = [{} for _ in texts]
            for cat in self.categories:
                pipe = self.pipelines.get(cat)
                fitted = False
                if pipe is not None:
                    try:
                        from sklearn.utils.validation import check_is_fitted
                        check_is_fitted(pipe)
                        probs = pipe.predict_proba(texts)[:, 1]
                        for idx, p in enumerate(probs):
                            results[idx][cat] = round(float(p), 4)
                        fitted = True
                    except Exception:
                        fitted = False
                if not fitted:
                    for idx, t in enumerate(texts):
                        results[idx][cat] = round(self._heuristic_fallback(t, cat), 4)
            return results

        return [self.score(t, surfaces[i] if surfaces else None) for i, t in enumerate(texts)]

    def _heuristic_fallback(self, text: str, category: str) -> float:
        """
        Heuristic fallback when unfitted, simulating higher sensitivity/recall characteristic of Family B.
        """
        t = text.lower()
        if category in ("Jailbreak", "Prompt Injection"):
            triggers = [
                "ignore previous", "ignore all", "instructions", "system prompt",
                "pretend", "jailbreak", "unrestricted", "bypass", "override",
                "hidden instruction", "assistant instructions", "developer mode", "do anything now"
            ]
            matches = sum(1 for trig in triggers if trig in t)
            if matches > 0:
                return min(1.0, 0.50 + 0.20 * matches)
            return 0.04
        elif category == "PII Leakage":
            triggers = ["ssn", "social security", "credit card", "private key", "api key", "password", "auth token", "secret key"]
            if any(trig in t for trig in triggers):
                return 0.80
            return 0.02
        elif category == "Malicious Tools":
            triggers = ["bash -i", "rm -rf", "drop database", "chmod 777", "powershell -enc", "wget http", "curl -s", "reverse shell"]
            if any(trig in t for trig in triggers):
                return 0.85
            return 0.03
        elif category == "Hate/Toxicity":
            triggers = ["kill all", "hate speech", "racial slur", "terrorist attack", "exterminate"]
            if any(trig in t for trig in triggers):
                return 0.80
            return 0.02
        return 0.04

    def save(self, directory: str) -> None:
        """
        Serializes all fitted MLP pipelines to disk using joblib.
        Creates one file per category: {directory}/family_b_{category}.pkl
        """
        import os
        import joblib
        os.makedirs(directory, exist_ok=True)
        for cat, pipe in self.pipelines.items():
            safe_name = cat.replace("/", "_").replace(" ", "_")
            path = os.path.join(directory, f"family_b_{safe_name}.pkl")
            joblib.dump(pipe, path)
        print(f"[CipherGuard] ModelFamilyB saved {len(self.pipelines)} pipelines to '{directory}'.")

    @classmethod
    def load(cls, directory: str, categories: list = None) -> "ModelFamilyB":
        """
        Restores fitted MLP pipelines from disk.
        Returns a ModelFamilyB instance with is_fitted=True.
        """
        import os
        import joblib
        instance = cls(categories=categories)
        loaded = 0
        for cat in instance.categories:
            safe_name = cat.replace("/", "_").replace(" ", "_")
            path = os.path.join(directory, f"family_b_{safe_name}.pkl")
            if os.path.exists(path):
                instance.pipelines[cat] = joblib.load(path)
                loaded += 1
        if loaded > 0:
            instance.is_fitted = True
        print(f"[CipherGuard] ModelFamilyB loaded {loaded}/{len(instance.categories)} pipelines from '{directory}'.")
        return instance
