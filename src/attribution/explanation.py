"""
Contrastive Explanation Formatter and Minimality Pruner.
Matches Protocol §6.2 and formats results for audit logging and UI display.
"""

from typing import List, Dict, Any, Optional, Callable
from .beam_search import BeamSearchResult


class ContrastiveExplanation:
    def __init__(
        self,
        is_flipped: bool,
        explanation_text: str,
        removed_tokens: List[str],
        cardinality: int,
        delta_score: float,
        original_score: float,
        flipped_score: float,
        sanitized_text: str,
        search_latency_ms: float,
        is_locally_minimal: bool = True
    ):
        self.is_flipped = is_flipped
        self.explanation_text = explanation_text
        self.removed_tokens = removed_tokens
        self.cardinality = cardinality
        self.delta_score = delta_score
        self.original_score = original_score
        self.flipped_score = flipped_score
        self.sanitized_text = sanitized_text
        self.search_latency_ms = search_latency_ms
        self.is_locally_minimal = is_locally_minimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_flipped": self.is_flipped,
            "explanation_text": self.explanation_text,
            "removed_tokens": self.removed_tokens,
            "cardinality": self.cardinality,
            "delta_score": round(self.delta_score, 4),
            "original_score": round(self.original_score, 4),
            "flipped_score": round(self.flipped_score, 4),
            "sanitized_text": self.sanitized_text,
            "search_latency_ms": round(self.search_latency_ms, 2),
            "is_locally_minimal": self.is_locally_minimal
        }


def format_contrastive_explanation(
    result: BeamSearchResult,
    original_text: str,
    score_fn: Optional[Callable[[List[str]], List[float]]] = None,
    threshold: Optional[float] = None
) -> ContrastiveExplanation:
    """
    Constructs a human-readable contrastive explanation and checks local minimality.
    """
    if not result.flipped:
        explanation = (
            f"Input blocked (score: {result.original_score:.3f}). "
            f"No minimal token subset could flip the decision below threshold "
            f"within search budget (beam={result.beam_width}, max_removals={result.removal_cap})."
        )
        return ContrastiveExplanation(
            is_flipped=False,
            explanation_text=explanation,
            removed_tokens=result.removed_tokens,
            cardinality=len(result.removed_tokens),
            delta_score=result.original_score - result.final_score,
            original_score=result.original_score,
            flipped_score=result.final_score,
            sanitized_text=result.remaining_text,
            search_latency_ms=result.search_latency_ms,
            is_locally_minimal=False
        )

    # If flipped with 0 removals (already below threshold)
    if not result.removed_tokens:
        return ContrastiveExplanation(
            is_flipped=True,
            explanation_text="Input is within safe threshold boundaries; no token removal necessary.",
            removed_tokens=[],
            cardinality=0,
            delta_score=0.0,
            original_score=result.original_score,
            flipped_score=result.final_score,
            sanitized_text=original_text,
            search_latency_ms=result.search_latency_ms,
            is_locally_minimal=True
        )

    # Verify local minimality if score_fn and threshold provided
    is_minimal = True
    active_tokens = list(result.removed_tokens)
    if score_fn is not None and threshold is not None and len(active_tokens) > 1:
        # Check if any single token removal was redundant
        for token_to_drop in list(active_tokens):
            test_removed = [t for t in active_tokens if t != token_to_drop]
            words = original_text.split()
            # Reconstruct sentence omitting test_removed
            rem_words = list(words)
            for t in test_removed:
                if t in rem_words:
                    rem_words.remove(t)
            trial_text = " ".join(rem_words)
            sub_score = score_fn([trial_text])[0]
            if sub_score < threshold:
                # The token was not strictly necessary for the flip!
                is_minimal = False
                break

    tokens_str = ", ".join([f"'{t}'" for t in result.removed_tokens])
    delta = result.original_score - result.final_score
    explanation = (
        f"Input blocked because of token subset: [{tokens_str}]. "
        f"Removing these {len(result.removed_tokens)} token(s) drops the risk score from "
        f"{result.original_score:.3f} to {result.final_score:.3f} (Δ={delta:.3f}), flipping the decision to PASS."
    )

    return ContrastiveExplanation(
        is_flipped=True,
        explanation_text=explanation,
        removed_tokens=result.removed_tokens,
        cardinality=len(result.removed_tokens),
        delta_score=delta,
        original_score=result.original_score,
        flipped_score=result.final_score,
        sanitized_text=result.remaining_text,
        search_latency_ms=result.search_latency_ms,
        is_locally_minimal=is_minimal
    )
