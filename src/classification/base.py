"""
Base Classifier Interface for CipherGuard
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseClassifier(ABC):
    """
    Abstract interface for risk classification models.
    Each classifier must produce calibrated risk scores s in [0, 1] per category.
    """

    @abstractmethod
    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """
        Score a single input string and return a dictionary of category -> risk score in [0, 1].
        """
        pass

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Score a batch of texts. Default implementation loops; override for vectorized batching.
        """
        if surfaces is None:
            surfaces = [None] * len(texts)
        return [self.score(t, s) for t, s in zip(texts, surfaces)]
