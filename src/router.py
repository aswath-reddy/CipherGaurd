"""
CipherGuard Core Router: Orchestrates Dual-Surface Ingestion, Classification,
Dynamic Policy Resolution, Contrastive Attribution, and Audit Logging.
Matches Protocol §2, §4, §5.
"""

import time
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field

from .ingestion import DirectInput, IndirectInput
from .classification import RiskAggregator, BaseClassifier
from .policy import PolicyLoader, DecisionEngine, RoutingAction
from .attribution import run_removal_beam_search, format_contrastive_explanation, ContrastiveExplanation
from .logging_store import AuditLogger


class RouterOutput(BaseModel):
    action: RoutingAction
    surface: str
    risk_scores: Dict[str, float]
    raw_scores: Dict[str, Dict[str, float]]
    triggered_rules: List[Dict[str, Any]]
    rationale: str
    explanation: Optional[Dict[str, Any]] = None
    sanitized_text: Optional[str] = None
    latency_ms: float
    audit_id: Optional[int] = None


class CipherGuardRouter:
    """
    Main Gateway Router for CipherGuard.
    """

    def __init__(
        self,
        policy_path: str = "policy/policy.json",
        risk_aggregator: Optional[RiskAggregator] = None,
        db_path: str = "cipherguard_audit.db",
        default_beam_width: int = 5,
        default_removal_cap: int = 6
    ):
        self.policy_loader = PolicyLoader(policy_path)
        self.decision_engine = DecisionEngine(self.policy_loader)
        self.risk_aggregator = risk_aggregator or RiskAggregator()
        self.audit_logger = AuditLogger(db_path)
        self.default_beam_width = default_beam_width
        self.default_removal_cap = default_removal_cap

    def route(
        self,
        input_data: Union[DirectInput, IndirectInput, str],
        surface: Optional[str] = None,
        beam_width: Optional[int] = None,
        removal_cap: Optional[int] = None,
        enable_attribution: bool = True
    ) -> RouterOutput:
        """
        Executes end-to-end routing pipeline.
        """
        start_time = time.perf_counter()

        # Step 1: Ingestion & surface resolution
        if isinstance(input_data, DirectInput):
            text = input_data.get_eval_text()
            surf = input_data.surface
        elif isinstance(input_data, IndirectInput):
            text = input_data.get_eval_text()
            surf = input_data.surface
        else:
            text = str(input_data).strip()
            surf = surface or "direct"

        # Step 2: Classification Layer
        agg_result = self.risk_aggregator.aggregate(text, surf)
        fused_scores = agg_result["fused_scores"]
        raw_scores = agg_result["raw_scores"]

        # Step 3: Policy & Decision Layer
        decision = self.decision_engine.evaluate(fused_scores, surf)

        # Step 4: Attribution Layer (if input blocked or flagged for review)
        explanation_dict = None
        sanitized_text = text

        if enable_attribution and decision.action in (RoutingAction.BLOCK, RoutingAction.REVIEW):
            bw = beam_width or self.default_beam_width
            cap = removal_cap or self.default_removal_cap

            # Determine dominant triggered category and threshold
            target_cat = "Prompt Injection"
            target_thresh = 0.45
            if decision.triggered_rules:
                top_rule = decision.triggered_rules[0]
                target_cat = top_rule["category"]
                target_thresh = top_rule["threshold"]

            # Define batched scoring function for the attribution beam search
            def batch_score_fn(candidate_texts: List[str]) -> List[float]:
                batch_agg = self.risk_aggregator.aggregate_batch(candidate_texts, [surf] * len(candidate_texts))
                return [b["fused_scores"].get(target_cat, 0.0) for b in batch_agg]

            beam_res = run_removal_beam_search(
                text=text,
                score_fn=batch_score_fn,
                threshold=target_thresh,
                beam_width=bw,
                removal_cap=cap
            )

            explanation = format_contrastive_explanation(
                result=beam_res,
                original_text=text,
                score_fn=batch_score_fn,
                threshold=target_thresh
            )
            explanation_dict = explanation.to_dict()
            sanitized_text = explanation.sanitized_text

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Audit Logging
        audit_id = self.audit_logger.log_decision(
            surface=surf,
            input_text=text,
            risk_scores=fused_scores,
            action=decision.action.value,
            rationale=decision.rationale,
            explanation=explanation_dict,
            latency_ms=total_latency_ms
        )

        return RouterOutput(
            action=decision.action,
            surface=surf,
            risk_scores=fused_scores,
            raw_scores=raw_scores,
            triggered_rules=decision.triggered_rules,
            rationale=decision.rationale,
            explanation=explanation_dict,
            sanitized_text=sanitized_text,
            latency_ms=round(total_latency_ms, 2),
            audit_id=audit_id
        )
