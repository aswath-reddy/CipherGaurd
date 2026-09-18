"""
Expanded Dataset Builder for CipherGuard — Phase 1 (Multi-label, Enriched).

Builds the complete multi-label dataset by combining:
  1. pilot_dataset_85.json       — Proof-of-concept dataset (n=85), converted to
                                   multi-label format. NEVER modified in place.
  2. deepset/prompt-injections   — HuggingFace public dataset (~350 samples)
  3. tweet_eval/hate             — HuggingFace public dataset (~200 samples)
  4. cipherguard_pii_synthetic   — Hand-crafted PII Leakage dataset (~99 samples)
  5. cipherguard_indirect_synthetic — Indirect surface attacks (~100 samples)
  6. cipherguard_maltools_synthetic — Improved Malicious Tools dataset (~200 malicious
                                      + ~200 benign = ~400 samples)

DATASET ROLES
─────────────
  pilot_dataset_85.json  → "Pilot / proof-of-concept dataset (n=85)"
                           NOT used for final training or evaluation.
                           See data/PILOT_DATASET_README.md.

  expanded_dataset.json  → Full combined dataset BEFORE splitting.
                           Used as input to split_dataset.py.

  data/train.json        → Training split (70%)  — produced by split_dataset.py
  data/val.json          → Validation split (15%) — produced by split_dataset.py
  data/test.json         → Final held-out test split (15%) — produced by split_dataset.py

OUTPUT SCHEMA (per sample)
──────────────────────────
  {
    "id":          int,
    "text":        str,
    "labels":      {"Jailbreak":0|1, "Prompt Injection":0|1,
                    "PII Leakage":0|1, "Malicious Tools":0|1, "Hate/Toxicity":0|1},
    "surface":     "direct" | "indirect",
    "attack_type": str,
    "severity":    "low" | "medium" | "high" | "critical",
    "source":      str,
    "annotation_tier":  int | None,   # PII samples only
    "annotation_note":  str | None,   # PII samples only
  }

RANDOM SEED
───────────
  RANDOM_SEED = 42  (all shuffling and augmentation)

Run:
    python -m evaluation.build_expanded_dataset

"""

import hashlib
import json
import os
import random
import sys
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
CATEGORIES: List[str] = [
    "Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"
]

# Allow running from project root or evaluation/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _resolve(relative_path: str) -> str:
    """Resolve a path relative to project root."""
    return os.path.join(_PROJECT_ROOT, relative_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def zero_labels(**kwargs) -> Dict[str, int]:
    base = {cat: 0 for cat in CATEGORIES}
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Pilot dataset integrity check
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pilot_integrity(pilot_path: str) -> str:
    """
    Verifies that pilot_dataset_85.json has not been modified since its checksum
    was first recorded. On first run, writes the checksum to data/.pilot_checksum.
    Raises RuntimeError if the checksum changes on subsequent runs.

    This enforces the invariant: the Pilot / proof-of-concept dataset (n=85)
    is NEVER modified by any pipeline step.
    """
    checksum_path = os.path.join(os.path.dirname(pilot_path), ".pilot_checksum")
    current = _compute_sha256(pilot_path)

    if os.path.exists(checksum_path):
        with open(checksum_path, "r") as f:
            recorded = f.read().strip()
        if current != recorded:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"INTEGRITY ERROR: pilot_dataset_85.json has been modified!\n"
                f"  Recorded SHA-256: {recorded}\n"
                f"  Current  SHA-256: {current}\n"
                f"The Pilot / proof-of-concept dataset (n=85) must NEVER be\n"
                f"altered. Restore it from version control before proceeding.\n"
                f"{'='*60}"
            )
        print(f"[Pilot] Integrity verified. SHA-256: {current[:16]}...")
    else:
        with open(checksum_path, "w") as f:
            f.write(current)
        print(f"[Pilot] First run — checksum recorded. SHA-256: {current[:16]}...")

    return current


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert pilot dataset (n=85) — READ-ONLY
# ─────────────────────────────────────────────────────────────────────────────

def load_and_convert_pilot(path: str) -> List[Dict[str, Any]]:
    """
    Converts pilot_dataset_85.json (binary labels) to multi-label format.

    Pilot dataset role: Proof-of-concept / feasibility demonstration (n=85).
    This is NOT the training or evaluation dataset.
    See data/PILOT_DATASET_README.md for full explanation.

    Conversion rules:
      - Direct surface, label=1 → Jailbreak=1, Prompt Injection=1
      - Indirect surface, label=1 → Prompt Injection=1 (RAG-injection pattern)
      - label=0 → all labels 0
    """
    verify_pilot_integrity(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    converted = []
    for d in raw:
        label_val = d.get("label", 0)
        surface = d.get("surface", "direct")
        labels = zero_labels()
        if label_val == 1:
            if surface == "direct":
                labels["Jailbreak"] = 1
                labels["Prompt Injection"] = 1
            else:
                labels["Prompt Injection"] = 1
        converted.append({
            "id": d["id"],
            "surface": surface,
            "text": d["text"],
            "labels": labels,
            "source": "pilot",
        })
    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: HuggingFace public datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_prompt_injections(max_samples: int = 350) -> List[Dict[str, Any]]:
    """
    Downloads deepset/prompt-injections from HuggingFace.
    Falls back gracefully if datasets library is not available — in that case
    returns an empty list and the caller should use load_cached_hf_samples().
    """
    try:
        from datasets import load_dataset
        print("[Dataset] Downloading deepset/prompt-injections...")
        ds = load_dataset("deepset/prompt-injections", split="train")
        samples = []
        seen_texts = set()
        for row in ds:
            text = row.get("text", "").strip()
            label = int(row.get("label", 0))
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            labels = zero_labels()
            if label == 1:
                labels["Jailbreak"] = 1
                labels["Prompt Injection"] = 1
            samples.append({
                "surface": "direct",
                "text": text,
                "labels": labels,
                "source": "deepset/prompt-injections",
            })
            if len(samples) >= max_samples:
                break
        print(f"[Dataset] Loaded {len(samples)} samples from deepset/prompt-injections.")
        return samples
    except Exception as e:
        print(f"[Dataset] Could not download deepset/prompt-injections: {e}")
        print("  Install with: pip install datasets")
        return []


def download_hate_speech(max_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Downloads tweet_eval/hate from HuggingFace.
    Balances hate vs non-hate at max_samples//2 each.
    Falls back gracefully if datasets library is not available.
    """
    try:
        from datasets import load_dataset
        print("[Dataset] Downloading tweet_eval (hate split)...")
        ds = load_dataset("tweet_eval", "hate", split="train")
        samples, seen_texts = [], set()
        benign_count = hate_count = 0
        for row in ds:
            text = row.get("text", "").strip()
            label = int(row.get("label", 0))
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            labels = zero_labels()
            is_hate = (label == 1)
            if is_hate:
                if hate_count >= max_samples // 2:
                    continue
                labels["Hate/Toxicity"] = 1
                hate_count += 1
            else:
                if benign_count >= max_samples // 2:
                    continue
                benign_count += 1
            samples.append({
                "surface": "direct",
                "text": text,
                "labels": labels,
                "source": "tweet_eval/hate",
            })
            if len(samples) >= max_samples:
                break
        print(f"[Dataset] Loaded {len(samples)} from tweet_eval/hate ({hate_count} hate, {benign_count} benign).")
        return samples
    except Exception as e:
        print(f"[Dataset] Could not download tweet_eval/hate: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: CipherGuard synthetic sources
# ─────────────────────────────────────────────────────────────────────────────

def load_pii_dataset() -> List[Dict[str, Any]]:
    """Loads PII Leakage samples from evaluation/pii_dataset.py."""
    try:
        from evaluation.pii_dataset import generate_pii_dataset
        return generate_pii_dataset()
    except ImportError:
        # Try relative import fallback
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pii_dataset", os.path.join(_HERE, "pii_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.generate_pii_dataset()


def load_indirect_surface_dataset() -> List[Dict[str, Any]]:
    """Loads indirect surface injection samples."""
    try:
        from evaluation.indirect_surface_dataset import generate_indirect_surface_dataset
        return generate_indirect_surface_dataset()
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "indirect_surface_dataset",
            os.path.join(_HERE, "indirect_surface_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.generate_indirect_surface_dataset()


def load_malicious_tools_dataset(rng: random.Random) -> List[Dict[str, Any]]:
    """Loads improved Malicious Tools samples."""
    try:
        from evaluation.malicious_tools_dataset import generate_improved_malicious_tools
        return generate_improved_malicious_tools(rng)
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "malicious_tools_dataset",
            os.path.join(_HERE, "malicious_tools_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.generate_improved_malicious_tools(rng)


def load_cached_hf_samples(
    output_path: str,
    sources: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loads previously cached HuggingFace samples from an existing expanded_dataset.json.

    When the HuggingFace `datasets` library is not available (e.g., offline or
    restricted environments), this function re-uses the pre-built samples from a
    prior run of the builder. It filters by source so that only HuggingFace-origin
    records are reused — the CipherGuard synthetic sources are always regenerated fresh.

    Args:
        output_path: Path to the existing expanded_dataset.json (from a prior run).
        sources: List of source strings to include. Defaults to HuggingFace sources.

    Returns:
        Filtered list of samples, or empty list if file not found.
    """
    if sources is None:
        sources = ["deepset/prompt-injections", "tweet_eval/hate"]

    # Look for the existing file before we overwrite it
    cache_path = output_path
    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Filter to only the requested sources
        result = [s for s in cached if s.get("source") in sources]
        # Ensure they have the new required fields (labels dict must be present)
        valid = []
        for s in result:
            if "labels" in s and "text" in s:
                # Drop old fields that will be re-enriched
                s.pop("attack_type", None)
                s.pop("severity", None)
                valid.append(s)
        if valid:
            print(f"[Cache] Loaded {len(valid)} cached HuggingFace samples from {cache_path}")
            src_counts = {}
            for s in valid:
                src = s.get("source", "?")
                src_counts[src] = src_counts.get(src, 0) + 1
            for src, cnt in src_counts.items():
                print(f"  {src}: {cnt}")
        return valid
    except Exception as e:
        print(f"[Cache] Could not load cached HuggingFace samples: {e}")
        return []



# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes exact-text duplicates. Keeps the first occurrence.
    Reports number of duplicates removed.
    """
    seen = set()
    deduped = []
    dup_count = 0
    for s in samples:
        t = s["text"].strip().lower()
        if t in seen:
            dup_count += 1
            continue
        seen.add(t)
        deduped.append(s)
    if dup_count:
        print(f"[Dedup] Removed {dup_count} exact duplicate texts. {len(deduped)} remain.")
    else:
        print(f"[Dedup] No exact duplicates found. {len(deduped)} samples.")
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Enrich metadata
# ─────────────────────────────────────────────────────────────────────────────

def enrich_all(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backfills attack_type and severity on all samples."""
    try:
        from evaluation.enrich_metadata import enrich_dataset
        return enrich_dataset(samples)
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "enrich_metadata", os.path.join(_HERE, "enrich_metadata.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enrich_dataset(samples)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Assemble and save
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save(
    pilot_path: str = "data/pilot_dataset_85.json",
    output_path: str = "data/expanded_dataset.json",
    max_injection_samples: int = 350,
    max_hate_samples: int = 200,
) -> str:
    """
    Builds the full expanded dataset and saves it to output_path.
    Returns output_path.

    Dataset roles:
      pilot_dataset_85.json  → Pilot / proof-of-concept (n=85). Read-only. Never modified.
      expanded_dataset.json  → Full combined dataset. Input to split_dataset.py.
      train/val/test.json    → Split outputs (produced by split_dataset.py separately).
    """
    rng = random.Random(RANDOM_SEED)

    print("=" * 65)
    print("CipherGuard Expanded Dataset Builder — Phase 1")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 65)

    pilot_path = _resolve(pilot_path)
    output_path = _resolve(output_path)

    # Pre-load cached HuggingFace samples from the EXISTING expanded_dataset.json
    # BEFORE we write anything new. This ensures the cache is available even when
    # the output file path is the same as the input (in-place rebuild).
    print("\n[Pre-flight] Loading cached HuggingFace samples from existing dataset...")
    cached_hf = load_cached_hf_samples(output_path)
    if cached_hf:
        print(f"  -> {len(cached_hf)} cached samples available as fallback.")
    else:
        print("  -> No existing cache found (first run or new output path).")

    all_samples: List[Dict[str, Any]] = []

    # 1. Pilot dataset (read-only, integrity-checked)
    print("\n[1/6] Converting pilot dataset (proof-of-concept, n=85)...")
    pilot = load_and_convert_pilot(pilot_path)
    all_samples.extend(pilot)
    print(f"  -> {len(pilot)} samples from pilot dataset.")

    # 2. Public prompt injection dataset
    print("\n[2/6] Downloading deepset/prompt-injections...")
    injections = download_prompt_injections(max_injection_samples)
    if not injections:
        print("  -> datasets unavailable; using pre-loaded cached HuggingFace samples...")
        injections = [s for s in cached_hf if s.get("source") == "deepset/prompt-injections"]
        print(f"  -> {len(injections)} samples from cache (deepset/prompt-injections).")
    else:
        print(f"  -> {len(injections)} samples.")
    all_samples.extend(injections)

    # 3. Hate speech dataset
    print("\n[3/6] Downloading tweet_eval/hate...")
    hate = download_hate_speech(max_hate_samples)
    if not hate:
        print("  -> datasets unavailable; using pre-loaded cached HuggingFace samples...")
        hate = [s for s in cached_hf if s.get("source") == "tweet_eval/hate"]
        print(f"  -> {len(hate)} samples from cache (tweet_eval/hate).")
    else:
        print(f"  -> {len(hate)} samples.")
    all_samples.extend(hate)

    # 4. PII Leakage synthetic dataset
    print("\n[4/6] Loading CipherGuard PII Leakage synthetic dataset...")
    pii = load_pii_dataset()
    all_samples.extend(pii)
    print(f"  -> {len(pii)} samples.")

    # 5. Indirect surface synthetic dataset
    print("\n[5/6] Loading CipherGuard indirect surface dataset...")
    indirect = load_indirect_surface_dataset()
    all_samples.extend(indirect)
    print(f"  -> {len(indirect)} samples.")

    # 6. Improved Malicious Tools
    print("\n[6/6] Loading CipherGuard improved Malicious Tools dataset...")
    maltools = load_malicious_tools_dataset(rng)
    all_samples.extend(maltools)
    print(f"  -> {len(maltools)} samples.")

    # Deduplication
    print("\n[Post-processing] Deduplicating...")
    all_samples = deduplicate(all_samples)

    # Metadata enrichment
    print("\n[Post-processing] Enriching metadata (attack_type, severity)...")
    all_samples = enrich_all(all_samples)

    # Assign sequential IDs and shuffle
    rng.shuffle(all_samples)
    for i, s in enumerate(all_samples, start=1):
        s["id"] = i

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"EXPANDED DATASET SUMMARY (saved to: {output_path})")
    print("=" * 65)
    print(f"Total samples: {len(all_samples)}")

    # Per source
    source_counts: Dict[str, int] = {}
    for s in all_samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    print("\nSamples per source:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src:<40}: {cnt:>5}")

    # Per category (multi-label — counts overlap)
    print("\nPer-category label counts (multi-label; counts overlap):")
    for cat in CATEGORIES:
        pos = sum(1 for s in all_samples if s["labels"].get(cat, 0) == 1)
        neg = len(all_samples) - pos
        print(f"  {cat:<25}: {pos:>5} positive | {neg:>5} negative")

    # Surface distribution
    direct_c = sum(1 for s in all_samples if s.get("surface") == "direct")
    indirect_c = sum(1 for s in all_samples if s.get("surface") == "indirect")
    print(f"\nSurface: direct={direct_c}, indirect={indirect_c}")

    # Severity
    sev_counts: Dict[str, int] = {}
    for s in all_samples:
        sv = s.get("severity", "unknown")
        sev_counts[sv] = sev_counts.get(sv, 0) + 1
    print("\nSeverity distribution:")
    for sv in ["low", "medium", "high", "critical", "unknown"]:
        if sv in sev_counts:
            print(f"  {sv:<10}: {sev_counts[sv]}")

    print(f"\nRandom seed used: {RANDOM_SEED}")
    print("\nNEXT STEP: Run split_dataset.py to produce train/val/test splits.")
    print("  python -m evaluation.split_dataset")
    print("=" * 65)

    return output_path


if __name__ == "__main__":
    # Allow running from project root or evaluation/ directory
    os.chdir(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)
    build_and_save()
