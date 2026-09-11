"""
Decision Engine: Evaluates calibrated risk score vector against active policy matrix.
Produces routing decisions (BLOCK, REVIEW, ALLOW) with full explanation rationale.
"""

from typing import Dict, Any, List, Optional
from .schema import RoutingAction, PolicyRule, PolicyConfig
from .loader import PolicyLoader


class DecisionResult:
    def __init__(self, action: RoutingAction, triggered_rules: List[Dict[str, Any]], rationale: str):
        self.action = action
        self.triggered_rules = triggered_rules
        self.rationale = rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "triggered_rules": self.triggered_rules,
            "rationale": self.rationale
        }


class DecisionEngine:
    """
    Evaluates risk score vectors against the dynamic policy matrix.
    Supports surface-specific overrides ('direct', 'indirect', or wildcard '*').
    """

    def __init__(self, policy_loader: PolicyLoader):
        self.loader = policy_loader

    def evaluate(self, risk_scores: Dict[str, float], surface: str) -> DecisionResult:
        """
        Evaluates risk scores against the loaded policy rules.
        Surface: 'direct' or 'indirect'.
        Returns DecisionResult.
        """
        # Ensure latest policy is active if modified
        self.loader.check_and_reload()
        config: PolicyConfig = self.loader.config

        surface_norm = surface.strip().lower() if surface else "*"
        triggered: List[Dict[str, Any]] = []

        # Find matching rules
        for rule in config.policies:
            # Check surface match (exact match or wildcard)
            if rule.surface not in (surface_norm, "*"):
                continue

            score = risk_scores.get(rule.category)
            if score is not None and score >= rule.threshold:
                triggered.append({
                    "category": rule.category,
                    "surface": rule.surface,
                    "score": score,
                    "threshold": rule.threshold,
                    "action": rule.action
                })

        if not triggered:
            return DecisionResult(
                action=config.default_action,
                triggered_rules=[],
                rationale=f"All category risk scores below policy thresholds on surface '{surface_norm}'. Passed by default."
            )

        # Priority ordering: BLOCK > REVIEW > ALLOW
        has_block = any(t["action"] == RoutingAction.BLOCK for t in triggered)
        has_review = any(t["action"] == RoutingAction.REVIEW for t in triggered)

        if has_block:
            final_action = RoutingAction.BLOCK
            reasons = [f"{t['category']} score {t['score']:.2f} >= threshold {t['threshold']:.2f}" 
                       for t in triggered if t["action"] == RoutingAction.BLOCK]
            rationale = f"Blocked on surface '{surface_norm}' due to: " + "; ".join(reasons)
        elif has_review:
            final_action = RoutingAction.REVIEW
            reasons = [f"{t['category']} score {t['score']:.2f} >= threshold {t['threshold']:.2f}" 
                       for t in triggered if t["action"] == RoutingAction.REVIEW]
            rationale = f"Flagged for review on surface '{surface_norm}' due to: " + "; ".join(reasons)
        else:
            final_action = config.default_action
            rationale = f"Evaluated rules triggered default action {final_action.value}."

        return DecisionResult(
            action=final_action,
            triggered_rules=triggered,
            rationale=rationale
        )
