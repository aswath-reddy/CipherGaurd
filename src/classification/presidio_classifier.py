"""
Presidio PII Classifier: Wraps Microsoft Presidio AnalyzerEngine for production-grade PII detection.
Detects 50+ PII entity types including credit cards, SSNs, emails, phone numbers, crypto wallets, etc.
Scores only the 'PII Leakage' category; returns 0.0 for all other categories.
Provides keyword-heuristic fallback if presidio_analyzer is not installed.
"""

from typing import Dict, List, Optional
from .base import BaseClassifier

CATEGORY = "PII Leakage"

ALL_CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]

# High-confidence PII entity types that should raise a strong signal
HIGH_CONFIDENCE_ENTITIES = {
    "CREDIT_CARD", "US_SSN", "US_PASSPORT", "US_BANK_NUMBER",
    "IBAN_CODE", "CRYPTO", "MEDICAL_LICENSE", "US_DRIVER_LICENSE"
}

# Moderate-confidence entities
MODERATE_ENTITIES = {
    "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "URL",
    "NRP", "LOCATION", "DATE_TIME"
}

# Keyword fallback triggers (used when presidio is not available)
FALLBACK_TRIGGERS = [
    "ssn", "social security", "credit card", "private key", "api_key", "api key",
    "password", "auth token", "secret key", "bearer token", "access token",
    "passport number", "bank account", "routing number", "cvv", "pin number",
    "driver license", "medical record", "patient id", "iban", "swift code",
    "private_key", "secret_key", "aws_access_key"
]


class PresidioClassifier(BaseClassifier):
    """
    PII Leakage detector using Microsoft Presidio AnalyzerEngine.

    Replaces the keyword-trigger heuristic in ModelFamilyA/B for the PII Leakage category.
    Detects 50+ entity types with confidence-weighted scoring.

    Score formula:
      - If entities found: score = min(1.0, max_confidence * boost_factor)
      - boost_factor = 1.2 for high-confidence entity types, 1.0 for moderate
      - If no entities: score = 0.0
    """

    def __init__(self, device: Optional[str] = None, lazy_load: bool = True):
        self._analyzer = None
        self._is_loaded = False
        self._fallback_mode = False
        if not lazy_load:
            self._load_analyzer()

    def _load_analyzer(self):
        if self._is_loaded:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Try large model first, fall back to small for memory-constrained environments
            try:
                configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
                }
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
                self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
                print("[CipherGuard] Presidio loaded with en_core_web_lg (high-accuracy NER).")
            except Exception:
                # Fall back to small spaCy model
                try:
                    configuration = {
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
                    }
                    provider = NlpEngineProvider(nlp_configuration=configuration)
                    nlp_engine = provider.create_engine()
                    self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
                    print("[CipherGuard] Presidio loaded with en_core_web_sm (lightweight NER).")
                except Exception as e2:
                    # Last resort: use default Presidio engine (no spaCy)
                    self._analyzer = AnalyzerEngine()
                    print(f"[CipherGuard] Presidio loaded in default mode: {e2}")

            self._is_loaded = True
            self._fallback_mode = False
        except ImportError:
            print("[CipherGuard] presidio_analyzer not installed. PII detection using keyword fallback.")
            print("  Install with: pip install presidio-analyzer && python -m spacy download en_core_web_lg")
            self._fallback_mode = True
            self._is_loaded = True
        except Exception as e:
            print(f"[CipherGuard] Presidio failed to load: {e}. Using keyword fallback.")
            self._fallback_mode = True
            self._is_loaded = True

    def _score_with_presidio(self, text: str) -> float:
        """Run Presidio analysis and return a confidence-weighted PII score."""
        try:
            results = self._analyzer.analyze(text=text, language="en")
            if not results:
                return 0.0

            best_score = 0.0
            for result in results:
                confidence = result.score
                entity_type = result.entity_type

                # Apply boost for high-confidence entity types
                if entity_type in HIGH_CONFIDENCE_ENTITIES:
                    boosted = min(1.0, confidence * 1.25)
                elif entity_type in MODERATE_ENTITIES:
                    boosted = confidence
                else:
                    boosted = confidence * 0.85  # Slight penalty for less critical types

                best_score = max(best_score, boosted)

            return round(best_score, 4)
        except Exception:
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> float:
        """Keyword-based PII heuristic (used when Presidio is unavailable)."""
        t = text.lower()
        hits = sum(1 for trigger in FALLBACK_TRIGGERS if trigger in t)
        if hits > 0:
            return min(1.0, 0.70 + 0.05 * (hits - 1))
        return 0.0

    def score(self, text: str, surface: Optional[str] = None) -> Dict[str, float]:
        """Score a single text. Only 'PII Leakage' is non-zero."""
        if not self._is_loaded:
            self._load_analyzer()

        base_scores = {cat: 0.0 for cat in ALL_CATEGORIES}

        if not text or not text.strip():
            return base_scores

        if self._fallback_mode:
            pii_score = self._keyword_fallback(text)
        else:
            pii_score = self._score_with_presidio(text)

        base_scores[CATEGORY] = pii_score
        return base_scores

    def score_batch(self, texts: List[str], surfaces: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """Batch PII scoring."""
        if not self._is_loaded:
            self._load_analyzer()
        return [self.score(t, surfaces[i] if surfaces else None) for i, t in enumerate(texts)]
