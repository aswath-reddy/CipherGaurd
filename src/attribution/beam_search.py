"""
Removal-Based Contrastive Token Attribution via Batched Beam Search.
Matches Protocol §6.2 and optimizes throughput with batched candidate scoring.
"""

import time
from typing import List, Tuple, Optional, Callable, Dict, Any


class BeamSearchResult:
    def __init__(
        self,
        flipped: bool,
        removed_tokens: List[str],
        final_score: float,
        original_score: float,
        steps_taken: int,
        search_latency_ms: float,
        remaining_text: str,
        beam_width: int,
        removal_cap: int
    ):
        self.flipped = flipped
        self.removed_tokens = removed_tokens
        self.final_score = final_score
        self.original_score = original_score
        self.steps_taken = steps_taken
        self.search_latency_ms = search_latency_ms
        self.remaining_text = remaining_text
        self.beam_width = beam_width
        self.removal_cap = removal_cap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flipped": self.flipped,
            "removed_tokens": self.removed_tokens,
            "cardinality": len(self.removed_tokens),
            "final_score": round(self.final_score, 4),
            "original_score": round(self.original_score, 4),
            "score_drop": round(self.original_score - self.final_score, 4),
            "steps_taken": self.steps_taken,
            "search_latency_ms": round(self.search_latency_ms, 2),
            "remaining_text": self.remaining_text,
            "budget": {
                "beam_width": self.beam_width,
                "removal_cap": self.removal_cap
            }
        }


def run_removal_beam_search(
    text: str,
    score_fn: Callable[[List[str]], List[float]],
    threshold: float,
    beam_width: int = 5,
    removal_cap: int = 6,
    tokenizer_fn: Optional[Callable[[str], List[str]]] = None
) -> BeamSearchResult:
    """
    Executes removal-based beam search over tokens to find a contrastive explanation.
    
    score_fn: takes a list of candidate strings and returns a list of risk scores in [0, 1].
    threshold: target risk score below which the input is considered flipped to PASS.
    beam_width: number of candidate branches retained at each removal step.
    removal_cap: maximum number of tokens allowed to be removed.
    """
    start_time = time.perf_counter()
    tokens = tokenizer_fn(text) if tokenizer_fn else text.split()

    if not tokens:
        return BeamSearchResult(
            flipped=False,
            removed_tokens=[],
            final_score=0.0,
            original_score=0.0,
            steps_taken=0,
            search_latency_ms=0.0,
            remaining_text="",
            beam_width=beam_width,
            removal_cap=removal_cap
        )

    # Initial score evaluation
    initial_scores = score_fn([text])
    original_score = initial_scores[0] if initial_scores else 0.0

    if original_score < threshold:
        # Already below threshold (not blocked)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return BeamSearchResult(
            flipped=True,
            removed_tokens=[],
            final_score=original_score,
            original_score=original_score,
            steps_taken=0,
            search_latency_ms=elapsed_ms,
            remaining_text=text,
            beam_width=beam_width,
            removal_cap=removal_cap
        )

    # State: list of tuples (remaining_tokens, removed_so_far)
    beams: List[Tuple[List[str], List[str]]] = [(tokens, [])]
    best_candidate_overall: Optional[Tuple[float, List[str], List[str]]] = None

    for step in range(removal_cap):
        candidates_to_score: List[Tuple[str, List[str], List[str]]] = []
        
        for remaining, removed in beams:
            for i in range(len(remaining)):
                trial = remaining[:i] + remaining[i+1:]
                trial_text = " ".join(trial)
                candidates_to_score.append((trial_text, trial, removed + [remaining[i]]))

        if not candidates_to_score:
            break

        # BATCHED SCORING: pass all candidate trial strings in a single batch
        trial_texts = [c[0] for c in candidates_to_score]
        scores = score_fn(trial_texts)

        # Pair scores with candidate metadata
        scored_candidates = []
        for s, (_, trial, rem) in zip(scores, candidates_to_score):
            scored_candidates.append((s, trial, rem))

        # Sort lowest risk score first (greedy objective: flip decision to PASS)
        scored_candidates.sort(key=lambda x: x[0])

        if best_candidate_overall is None or scored_candidates[0][0] < best_candidate_overall[0]:
            best_candidate_overall = scored_candidates[0]

        # Check if the top candidate successfully flipped below the threshold
        top_score, top_trial, top_removed = scored_candidates[0]
        if top_score < threshold:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return BeamSearchResult(
                flipped=True,
                removed_tokens=top_removed,
                final_score=top_score,
                original_score=original_score,
                steps_taken=step + 1,
                search_latency_ms=elapsed_ms,
                remaining_text=" ".join(top_trial),
                beam_width=beam_width,
                removal_cap=removal_cap
            )

        # Retain top-k beams for the next step
        # Deduplicate identical candidate remaining texts
        seen = set()
        next_beams = []
        for s, trial, rem in scored_candidates:
            key = " ".join(trial)
            if key not in seen:
                seen.add(key)
                next_beams.append((trial, rem))
            if len(next_beams) >= beam_width:
                break

        beams = next_beams

    # Exceeded budget without flipping
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    final_s = best_candidate_overall[0] if best_candidate_overall else original_score
    rem_tokens = best_candidate_overall[2] if best_candidate_overall else []
    rem_text = " ".join(best_candidate_overall[1]) if best_candidate_overall else text

    return BeamSearchResult(
        flipped=False,
        removed_tokens=rem_tokens,
        final_score=final_s,
        original_score=original_score,
        steps_taken=removal_cap,
        search_latency_ms=elapsed_ms,
        remaining_text=rem_text,
        beam_width=beam_width,
        removal_cap=removal_cap
    )
