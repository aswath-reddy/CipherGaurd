"""
Model Family A: Lightweight Linear Discriminator (TF-IDF + Logistic Regression / DistilBERT interface)
Matches Section IV-A and Protocol Family A specifications.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .base import BaseClassifier


class ModelFamilyA(BaseClassifier):
    """
    Family A classifier: Linear Discriminative Probe.
    Uses unigram & bigram TF-IDF with L2-regularized Logistic Regression.
    Provides fast, deterministic risk probabilities per category.
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
                ("clf", LogisticRegression(C=1.0, max_iter=200, class_weight="balanced"))
            ])

    def fit(self, texts: List[str], labels: Dict[str, List[int]]):
        """
        Fit a binary pipeline for each risk category.
        labels: Dict mapping category_name -> list of binary 0/1 indicators.
        """
        for cat, y in labels.items():
            if cat in self.pipelines:
                # Need at least two classes (0 and 1) to fit a binary classifier
                if len(set(y)) >= 2:
                    self.pipelines[cat].fit(texts, y)
                else:
                    print(f"[CipherGuard] ModelFamilyA: Category '{cat}' has only 1 class in training data. Keeping fallback heuristic.")
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
        Fast vectorized scoring for batches.
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
        Prior-based baseline heuristic if pipeline is unfitted before training.
        """
        t = text.lower()
        if category in ("Jailbreak", "Prompt Injection"):
            triggers = [
                "ignore previous instructions", "ignore all instructions", "system prompt",
                "pretend you are", "jailbreak", "unrestricted ai", "bypass safety",
                "override instructions", "hidden instruction", "dan mode"
            ]
            matches = sum(1 for trig in triggers if trig in t)
            if matches > 0:
                return min(1.0, 0.45 + 0.25 * matches)
            return 0.05
        elif category == "PII Leakage":
            triggers = ["ssn", "social security", "credit card", "private key", "password", "api_key"]
            if any(trig in t for trig in triggers):
                return 0.75
            return 0.02
        elif category == "Malicious Tools":
            triggers = ["bash -i", "rm -rf", "drop database", "chmod 777", "powershell -enc", "wget http"]
            if any(trig in t for trig in triggers):
                return 0.80
            return 0.03
        elif category == "Hate/Toxicity":
            triggers = ["kill all", "hate you", "threaten", "terrorist", "attack"]
            if any(trig in t for trig in triggers):
                return 0.75
            return 0.02
        return 0.05

    def save(self, directory: str) -> None:
        """
        Serializes all fitted pipelines to disk using joblib.
        Creates one file per category: {directory}/family_a_{category}.pkl
        """
        import os
        import joblib
        os.makedirs(directory, exist_ok=True)
        for cat, pipe in self.pipelines.items():
            safe_name = cat.replace("/", "_").replace(" ", "_")
            path = os.path.join(directory, f"family_a_{safe_name}.pkl")
            joblib.dump(pipe, path)
        print(f"[CipherGuard] ModelFamilyA saved {len(self.pipelines)} pipelines to '{directory}'.")

    @classmethod
    def load(cls, directory: str, categories: list = None) -> "ModelFamilyA":
        """
        Restores fitted pipelines from disk.
        Returns a ModelFamilyA instance with is_fitted=True.
        """
        import os
        import joblib
        instance = cls(categories=categories)
        loaded = 0
        for cat in instance.categories:
            safe_name = cat.replace("/", "_").replace(" ", "_")
            path = os.path.join(directory, f"family_a_{safe_name}.pkl")
            if os.path.exists(path):
                instance.pipelines[cat] = joblib.load(path)
                loaded += 1
        if loaded > 0:
            instance.is_fitted = True
        print(f"[CipherGuard] ModelFamilyA loaded {loaded}/{len(instance.categories)} pipelines from '{directory}'.")
        return instance
