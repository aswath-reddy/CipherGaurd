import json

with open("data/expanded_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Source counts
sources = {}
for d in data:
    s = d.get("source", "unknown")
    sources[s] = sources.get(s, 0) + 1
print("\nSamples per source:")
for k, v in sorted(sources.items()):
    print(f"  {k}: {v}")

# Per-category label counts
cats = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
print("\nPer-category label counts:")
for c in cats:
    pos = sum(1 for d in data if d.get("labels", {}).get(c, 0) == 1)
    neg = len(data) - pos
    print(f"  {c}: {pos} positive | {neg} negative")

# Surface counts
direct = sum(1 for d in data if d.get("surface") == "direct")
indirect = sum(1 for d in data if d.get("surface") == "indirect")
print(f"\nSurface: direct={direct}, indirect={indirect}")

# Check for attack_type and severity fields
has_attack_type = sum(1 for d in data if "attack_type" in d)
has_severity = sum(1 for d in data if "severity" in d)
print(f"\nRecords with attack_type: {has_attack_type}")
print(f"Records with severity: {has_severity}")
