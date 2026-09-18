"""
Phase 3 - Section 14: Policy Engine Unit Tests.

Tests the MODEL/POLICY separation: the decision engine maps risk scores to
routing actions without retraining or reloading model weights.

Coverage:
  1. Low/medium/high risk -> ALLOW/REVIEW/BLOCK
  2. Boundary values (thr-epsilon = ALLOW, thr+epsilon = BLOCK)
  3. Direct surface uses direct threshold, indirect uses separate indirect threshold
  4. Multi-category simultaneous trigger: BLOCK wins over REVIEW
  5. Policy JSON hot-reload: write new policy -> next call reads new thresholds
  6. REVIEW threshold routes to REVIEW not BLOCK
  7. Default action when nothing triggered = ALLOW
  8. Wildcard surface rule matches both direct and indirect
  9. Surface-specific rule does not match wrong surface
  10. Multi-rule: both BLOCK and REVIEW triggered -> BLOCK wins
  11. All categories clean -> ALLOW with rationale
  12. Score exactly equal to threshold -> triggers (>= semantics)
  13. Editing policy changes routing without model retraining
"""

import json
import os
import sys
import tempfile
import shutil

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.policy.loader import PolicyLoader
from src.policy.decision_engine import DecisionEngine
from src.policy.schema import RoutingAction

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_policy_dir(tmp_path):
    """Create a temp directory with a policy.json for hot-reload tests."""
    return tmp_path


def make_policy(tmp_path, policies, default="ALLOW"):
    policy = {"version": "test", "policies": policies, "default_action": default}
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    return str(p)


def engine_from_policy(policies, default="ALLOW", tmp_path=None):
    """Build an in-memory DecisionEngine from a policy dict."""
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
        cleanup = True
    else:
        cleanup = False
    path = make_policy(type("obj",(object,),{"__truediv__": lambda s,n: type("P",(object,),
        {"write_text": lambda s2,t: open(os.path.join(str(tmp_path),n),"w").write(t)})()})(),
        policies, default)
    # Simpler: just write directly
    import tempfile as tf
    fd, fpath = tf.mkstemp(suffix=".json")
    os.close(fd)
    with open(fpath, "w") as f:
        json.dump({"version":"test","policies":policies,"default_action":default}, f)
    loader = PolicyLoader(fpath)
    engine = DecisionEngine(loader)
    return engine, fpath


# ── Test 1: Low risk -> ALLOW ────────────────────────────────────────────────
def test_low_risk_allows():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "*", "threshold": 0.5, "action": "BLOCK"}
    ])
    scores = {cat: 0.1 for cat in CATEGORIES}
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.ALLOW
    assert result.triggered_rules == []


# ── Test 2: High risk -> BLOCK ───────────────────────────────────────────────
def test_high_risk_blocks():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "*", "threshold": 0.5, "action": "BLOCK"}
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = 0.9
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.BLOCK
    assert any(r["category"] == "Jailbreak" for r in result.triggered_rules)


# ── Test 3: Boundary - score just below threshold -> ALLOW ───────────────────
def test_boundary_below_threshold_allows():
    thr = 0.5
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "*", "threshold": thr, "action": "BLOCK"}
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = thr - 0.001
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.ALLOW


# ── Test 4: Boundary - score exactly at threshold -> BLOCK (>= semantics) ────
def test_boundary_at_threshold_blocks():
    thr = 0.5
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "*", "threshold": thr, "action": "BLOCK"}
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = thr
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.BLOCK


# ── Test 5: Direct surface uses direct threshold (not indirect) ───────────────
def test_surface_specific_direct_uses_direct_threshold():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "direct", "threshold": 0.4, "action": "BLOCK"},
        {"category": "Jailbreak", "surface": "indirect", "threshold": 0.8, "action": "BLOCK"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = 0.5   # above direct=0.4, below indirect=0.8
    direct_result   = engine.evaluate(scores, "direct")
    indirect_result = engine.evaluate(scores, "indirect")
    assert direct_result.action == RoutingAction.BLOCK   # triggers direct rule
    assert indirect_result.action == RoutingAction.ALLOW  # doesn't meet indirect threshold


# ── Test 6: Indirect surface uses indirect threshold (not direct) ─────────────
def test_surface_specific_indirect_uses_indirect_threshold():
    engine, _ = engine_from_policy([
        {"category": "Prompt Injection", "surface": "direct", "threshold": 0.8, "action": "BLOCK"},
        {"category": "Prompt Injection", "surface": "indirect", "threshold": 0.3, "action": "BLOCK"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Prompt Injection"] = 0.5  # below direct=0.8, above indirect=0.3
    direct_result   = engine.evaluate(scores, "direct")
    indirect_result = engine.evaluate(scores, "indirect")
    assert direct_result.action == RoutingAction.ALLOW
    assert indirect_result.action == RoutingAction.BLOCK


# ── Test 7: Multi-category trigger - BLOCK beats REVIEW ──────────────────────
def test_block_wins_over_review_when_both_triggered():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak",      "surface": "*", "threshold": 0.4, "action": "BLOCK"},
        {"category": "Hate/Toxicity",  "surface": "*", "threshold": 0.4, "action": "REVIEW"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"]     = 0.8
    scores["Hate/Toxicity"] = 0.8
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.BLOCK
    assert len(result.triggered_rules) == 2


# ── Test 8: REVIEW threshold -> REVIEW not BLOCK ─────────────────────────────
def test_review_threshold_routes_to_review():
    engine, _ = engine_from_policy([
        {"category": "Malicious Tools", "surface": "*", "threshold": 0.4, "action": "REVIEW"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Malicious Tools"] = 0.6
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.REVIEW


# ── Test 9: All below threshold -> ALLOW with rationale ──────────────────────
def test_all_below_threshold_allow_with_rationale():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "*", "threshold": 0.9, "action": "BLOCK"},
    ])
    scores = {cat: 0.1 for cat in CATEGORIES}
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.ALLOW
    assert "below" in result.rationale.lower() or "passed" in result.rationale.lower()


# ── Test 10: Wildcard surface matches both direct and indirect ────────────────
def test_wildcard_surface_matches_both():
    engine, _ = engine_from_policy([
        {"category": "PII Leakage", "surface": "*", "threshold": 0.5, "action": "BLOCK"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["PII Leakage"] = 0.7
    assert engine.evaluate(scores, "direct").action   == RoutingAction.BLOCK
    assert engine.evaluate(scores, "indirect").action == RoutingAction.BLOCK


# ── Test 11: Surface-specific rule does NOT match wrong surface ───────────────
def test_surface_specific_does_not_match_wrong_surface():
    engine, _ = engine_from_policy([
        {"category": "Jailbreak", "surface": "direct", "threshold": 0.3, "action": "BLOCK"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = 0.9
    result = engine.evaluate(scores, "indirect")  # indirect won't match direct-only rule
    assert result.action == RoutingAction.ALLOW


# ── Test 12: Policy JSON hot-reload changes routing without model retrain ─────
def test_policy_hot_reload_changes_routing(tmp_path):
    policy_path = str(tmp_path / "dynamic_policy.json")

    # Initial policy: high threshold (won't block)
    initial = {"version":"1","policies":[
        {"category":"Jailbreak","surface":"*","threshold":0.95,"action":"BLOCK"}
    ], "default_action":"ALLOW"}
    with open(policy_path, "w") as f:
        json.dump(initial, f)

    loader = PolicyLoader(policy_path)
    engine = DecisionEngine(loader)

    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Jailbreak"] = 0.7

    result_before = engine.evaluate(scores, "direct")
    assert result_before.action == RoutingAction.ALLOW   # 0.7 < 0.95

    # Rewrite policy file with lower threshold
    import time; time.sleep(0.05)  # ensure mtime changes
    updated = {"version":"1","policies":[
        {"category":"Jailbreak","surface":"*","threshold":0.5,"action":"BLOCK"}
    ], "default_action":"ALLOW"}
    with open(policy_path, "w") as f:
        json.dump(updated, f)

    # Force mtime to change (some filesystems have 1s resolution)
    os.utime(policy_path, None)
    loader._last_mtime = 0   # force reload on next check

    result_after = engine.evaluate(scores, "direct")
    assert result_after.action == RoutingAction.BLOCK    # 0.7 >= 0.5 after reload


# ── Test 13: Triggered rule contains correct category, score, threshold ───────
def test_triggered_rule_fields_complete():
    thr = 0.6
    engine, _ = engine_from_policy([
        {"category": "Hate/Toxicity", "surface": "*", "threshold": thr, "action": "BLOCK"},
    ])
    scores = {cat: 0.0 for cat in CATEGORIES}
    scores["Hate/Toxicity"] = 0.85
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.BLOCK
    rule = result.triggered_rules[0]
    assert rule["category"] == "Hate/Toxicity"
    assert abs(rule["score"] - 0.85) < 1e-6
    assert abs(rule["threshold"] - thr) < 1e-6
    assert rule["action"] == RoutingAction.BLOCK


# ── Test 14: Default action configurable ─────────────────────────────────────
def test_custom_default_action():
    engine, _ = engine_from_policy(
        [{"category":"Jailbreak","surface":"*","threshold":0.9,"action":"BLOCK"}],
        default="REVIEW"   # custom default
    )
    scores = {cat: 0.0 for cat in CATEGORIES}   # nothing triggered
    result = engine.evaluate(scores, "direct")
    assert result.action == RoutingAction.REVIEW  # falls back to custom default


if __name__ == "__main__":
    import pytest as pt
    pt.main([__file__, "-v"])
