"""Investigate the near-duplicate pairs found by check_leakage.py."""
import json, random

RANDOM_SEED = 42
THRESHOLD = 0.80
NGRAM_SIZE = 3

def ngrams(text, n=NGRAM_SIZE):
    t = " ".join(text.lower().split())
    if len(t) < n:
        return {t}
    return {t[i:i+n] for i in range(len(t) - n + 1)}

def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0

with open("data/train.json", encoding="utf-8") as f:
    train = json.load(f)
with open("data/val.json", encoding="utf-8") as f:
    val = json.load(f)
with open("data/test.json", encoding="utf-8") as f:
    test = json.load(f)

def find_suspicious(split_a, split_b, name_a, name_b, limit=20):
    rng = random.Random(RANDOM_SEED)
    ng_a = [(s, ngrams(s["text"])) for s in split_a]
    ng_b = [(s, ngrams(s["text"])) for s in split_b]
    suspicious = []
    # Exhaustive for small, sampled for large
    total = len(ng_a) * len(ng_b)
    if total <= 100000:
        pairs = [(a, b) for a in ng_a for b in ng_b]
    else:
        pairs = [(rng.choice(ng_a), rng.choice(ng_b)) for _ in range(100000)]
    for a, b in pairs:
        sim = jaccard(a[1], b[1])
        if sim >= THRESHOLD:
            suspicious.append((sim, a[0], b[0]))
    suspicious.sort(reverse=True)
    print(f"\n=== {name_a} x {name_b}: {len(suspicious)} suspicious pairs ===")
    for sim, sa, sb in suspicious[:limit]:
        print(f"\n  Jaccard={sim:.4f}")
        print(f"  [{name_a}] {sa['text'][:120]!r}")
        print(f"  [{name_b}] {sb['text'][:120]!r}")
        print(f"  Labels A: {sa['labels']}")
        print(f"  Labels B: {sb['labels']}")

find_suspicious(train, val, "train", "val")
find_suspicious(train, test, "train", "test")
