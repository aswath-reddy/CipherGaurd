"""
Unit Tests for Dynamic Policy Engine and Schema Validation.
Matches Protocol §7.
"""

import os
import json
import pytest
from src.policy import PolicyLoader, DecisionEngine, RoutingAction, PolicyConfig


@pytest.fixture
def temp_policy(tmp_path):
    policy_file = tmp_path / "test_policy.json"
    data = {
        "version": "1.0",
        "policies": [
            {"category": "Jailbreak", "surface": "direct", "threshold": 0.45, "action": "BLOCK"},
            {"category": "Jailbreak", "surface": "indirect", "threshold": 0.42, "action": "BLOCK"},
            {"category": "PII Leakage", "surface": "*", "threshold": 0.70, "action": "BLOCK"},
            {"category": "Malicious Tools", "surface": "*", "threshold": 0.65, "action": "REVIEW"}
        ],
        "default_action": "ALLOW"
    }
    with open(policy_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(policy_file)


def test_policy_load_and_resolve(temp_policy):
    loader = PolicyLoader(temp_policy)
    engine = DecisionEngine(loader)

    # Test safe input
    scores_safe = {"Jailbreak": 0.10, "PII Leakage": 0.05}
    res = engine.evaluate(scores_safe, surface="direct")
    assert res.action == RoutingAction.ALLOW

    # Test direct jailbreak block
    scores_unsafe_direct = {"Jailbreak": 0.50}
    res_direct = engine.evaluate(scores_unsafe_direct, surface="direct")
    assert res_direct.action == RoutingAction.BLOCK
    assert len(res_direct.triggered_rules) == 1

    # Test review action
    scores_review = {"Malicious Tools": 0.70}
    res_rev = engine.evaluate(scores_review, surface="direct")
    assert res_rev.action == RoutingAction.REVIEW


def test_surface_specific_thresholds(temp_policy):
    loader = PolicyLoader(temp_policy)
    engine = DecisionEngine(loader)

    # Score of 0.43:
    # On 'direct' surface, threshold is 0.45 -> should PASS (ALLOW)
    # On 'indirect' surface, threshold is 0.42 -> should BLOCK
    borderline_score = {"Jailbreak": 0.43}

    res_dir = engine.evaluate(borderline_score, surface="direct")
    assert res_dir.action == RoutingAction.ALLOW

    res_ind = engine.evaluate(borderline_score, surface="indirect")
    assert res_ind.action == RoutingAction.BLOCK


def test_fail_safe_on_malformed_policy(temp_policy):
    loader = PolicyLoader(temp_policy)
    old_version = loader.config.version

    # Write broken JSON to policy file
    with open(temp_policy, "w", encoding="utf-8") as f:
        f.write("{ INVALID JSON NOT PARSABLE }")

    # Reload should fail and keep active configuration safely
    success, msg = loader.reload()
    assert success is False
    assert "Failed to reload policy" in msg
    assert loader.config.version == old_version  # Fails safe, not open!
