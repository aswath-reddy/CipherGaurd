"""
Metadata Enrichment for CipherGuard Dataset.

Backfills `attack_type` and `severity` fields onto records that were ingested
from external sources (deepset/prompt-injections, tweet_eval/hate, pilot) and
do not already carry these fields.

Rules are heuristic and based on:
  - Label combination
  - Source identifier
  - Surface (direct vs indirect)
  - Text keyword signals (only for disambiguation, not sole classification)

This module is applied AFTER combining all sources, BEFORE splitting.
"""

from typing import Dict, Any, List


# ─────────────────────────────────────────────────────────────────────────────
# Attack type taxonomy
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_TYPE_MAP = {
    # (source, jailbreak, injection, pii, maltools, hate) → attack_type
    # Benign
    "benign": "benign",

    # Deepset prompt injections source — label=1 → jailbreak + prompt injection
    "deepset_positive": "jailbreak_prompt_injection",
    "deepset_negative": "benign",

    # Tweet eval / hate
    "tweet_hate_positive": "hate_speech",
    "tweet_hate_negative": "benign",

    # Pilot direct malicious → jailbreak + injection
    "pilot_direct_malicious": "jailbreak_prompt_injection",
    "pilot_indirect_malicious": "rag_injection",
    "pilot_benign": "benign",
}


def _classify_attack_type(sample: Dict[str, Any]) -> str:
    """
    Derive attack_type from label pattern and source.
    """
    # If already set, preserve it
    if sample.get("attack_type"):
        return sample["attack_type"]

    labels = sample.get("labels", {})
    source = sample.get("source", "")
    surface = sample.get("surface", "direct")

    jb = labels.get("Jailbreak", 0)
    pi = labels.get("Prompt Injection", 0)
    pii = labels.get("PII Leakage", 0)
    mt = labels.get("Malicious Tools", 0)
    ht = labels.get("Hate/Toxicity", 0)

    all_zero = not any([jb, pi, pii, mt, ht])

    if all_zero:
        return "benign"

    if ht and not any([jb, pi, pii, mt]):
        return "hate_speech"

    if pii and jb:
        return "credential_extraction_jailbreak"
    if pii and not jb:
        if source.startswith("cipherguard_pii"):
            return "pii_extraction"
        return "pii_in_context"

    if mt and not any([jb, pi]):
        return "malicious_tool"

    if jb and pi:
        if surface == "indirect":
            return "rag_injection"
        return "jailbreak_prompt_injection"

    if pi and surface == "indirect":
        return "rag_injection"

    if jb:
        return "jailbreak"

    if pi:
        return "prompt_injection"

    return "unknown"


def _classify_severity(sample: Dict[str, Any]) -> str:
    """
    Derive severity from label pattern and attack type.
    """
    # If already set, preserve it
    if sample.get("severity"):
        return sample["severity"]

    labels = sample.get("labels", {})
    attack_type = sample.get("attack_type", "benign")

    jb = labels.get("Jailbreak", 0)
    pi = labels.get("Prompt Injection", 0)
    pii = labels.get("PII Leakage", 0)
    mt = labels.get("Malicious Tools", 0)
    ht = labels.get("Hate/Toxicity", 0)

    all_zero = not any([jb, pi, pii, mt, ht])

    if all_zero:
        return "low"

    positive_count = jb + pi + pii + mt + ht

    # Credential extraction is always critical
    if "credential" in attack_type or "extraction" in attack_type:
        return "critical"

    # PII always at least high
    if pii:
        return "critical" if (jb or mt) else "high"

    # Malicious tools
    if mt:
        if "obfuscated" in attack_type or "reverse_shell" in attack_type or "malware" in attack_type:
            return "critical"
        if "sql_injection" in attack_type or "destructive" in attack_type or "exfiltration" in attack_type:
            return "critical"
        return "high"

    # Jailbreak + injection = high
    if jb and pi:
        return "high"

    # Pure jailbreak or injection
    if jb or pi:
        return "high" if positive_count >= 2 else "medium"

    # Hate/toxicity alone
    if ht:
        return "medium"

    return "low"


def enrich_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a copy of the sample with attack_type and severity populated.
    Does not modify the original.
    """
    enriched = dict(sample)
    enriched["attack_type"] = _classify_attack_type(enriched)
    enriched["severity"] = _classify_severity(enriched)
    return enriched


def enrich_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies enrich_sample to every record in the dataset.
    Records that already carry these fields have them preserved.

    Returns the enriched list (same length, same order).
    """
    enriched = [enrich_sample(s) for s in samples]

    # Report enrichment stats
    attack_types = {}
    severities = {}
    for s in enriched:
        at = s.get("attack_type", "unknown")
        sv = s.get("severity", "unknown")
        attack_types[at] = attack_types.get(at, 0) + 1
        severities[sv] = severities.get(sv, 0) + 1

    print("[Enrichment] Attack type distribution:")
    for k, v in sorted(attack_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("[Enrichment] Severity distribution:")
    for k, v in sorted(severities.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return enriched


if __name__ == "__main__":
    import json

    # Quick self-test
    test_samples = [
        {"labels": {"Jailbreak": 1, "Prompt Injection": 1, "PII Leakage": 0, "Malicious Tools": 0, "Hate/Toxicity": 0},
         "surface": "direct", "source": "deepset/prompt-injections"},
        {"labels": {"Jailbreak": 0, "Prompt Injection": 0, "PII Leakage": 0, "Malicious Tools": 0, "Hate/Toxicity": 1},
         "surface": "direct", "source": "tweet_eval/hate"},
        {"labels": {"Jailbreak": 0, "Prompt Injection": 0, "PII Leakage": 1, "Malicious Tools": 0, "Hate/Toxicity": 0},
         "surface": "direct", "source": "cipherguard_pii_synthetic"},
        {"labels": {"Jailbreak": 0, "Prompt Injection": 0, "PII Leakage": 0, "Malicious Tools": 0, "Hate/Toxicity": 0},
         "surface": "direct", "source": "pilot"},
    ]
    enriched = enrich_dataset(test_samples)
    print(json.dumps(enriched, indent=2))
