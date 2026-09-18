"""
Phase 3 - Section 16: Removal-Based Attribution Evaluation.

Measures:
  - Flip rate (fraction of inputs where beam search found a flipping subset)
  - Average tokens removed on success
  - Search latency (ms)
  - Success rate by attack_type and by surface (direct/indirect)
  - beam_width=5 vs beam_width=10 comparison (removal_cap fixed at 6)
  - Notes on GPU batching: score_fn for Strategy D is TF-IDF+LogReg/MLP
    (CPU-only), so GPU batching is NOT applicable. The score_fn already
    accepts batches (aggregate_batch) -- reported as batched CPU.

Outputs:
  evaluation/results/attribution_eval.json
"""

import os, sys, json, time, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from src.classification.model_family_a import ModelFamilyA
from src.classification.model_family_b import ModelFamilyB
from src.attribution.beam_search import run_removal_beam_search
from evaluation.build_dataset import get_category_labels

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
BLOCK_THR  = {"Jailbreak":0.45,"Prompt Injection":0.45,"PII Leakage":0.45,
               "Malicious Tools":0.45,"Hate/Toxicity":0.55}
RESULTS_DIR= "evaluation/results"
N_SAMPLE   = 30   # positives to sample
RANDOM_SEED= 42

print("=" * 65)
print("CipherGuard Phase 3 -- Section 16: Attribution Evaluation")
print("=" * 65)

mA = ModelFamilyA.load("models")
mB = ModelFamilyB.load("models")

# Load val split (never use test for tuning/eval here)
val_data = json.load(open("data/val.json", encoding="utf-8"))
texts_v  = [d["text"] for d in val_data]
surfaces_v = [d.get("surface","direct") for d in val_data]
atypes_v   = [d.get("attack_type","unknown") for d in val_data]

# Score all val samples with Strategy D
batch_a = mA.score_batch(texts_v, surfaces_v)
batch_b = mB.score_batch(texts_v, surfaces_v)
probs_v = {cat: [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(batch_a, batch_b)]
           for cat in CATEGORIES}

# Select confirmed positives: score >= BLOCK_THR for their dominant category
rng = np.random.default_rng(RANDOM_SEED)
positives = []
for i, (text, surf, atype) in enumerate(zip(texts_v, surfaces_v, atypes_v)):
    for cat in CATEGORIES:
        if probs_v[cat][i] >= BLOCK_THR[cat]:
            positives.append({
                "idx": i, "text": text, "surface": surf,
                "attack_type": atype, "category": cat,
                "score": probs_v[cat][i],
            })
            break  # one entry per sample

# Sample N_SAMPLE, ensuring surface diversity
direct_pos   = [p for p in positives if p["surface"]=="direct"]
indirect_pos = [p for p in positives if p["surface"]=="indirect"]
n_direct  = min(len(direct_pos),   N_SAMPLE * 2 // 3)
n_indirect= min(len(indirect_pos), N_SAMPLE - n_direct)
sampled = (rng.choice(direct_pos,   size=min(n_direct, len(direct_pos)),   replace=False).tolist() +
           rng.choice(indirect_pos, size=min(n_indirect,len(indirect_pos)), replace=False).tolist())
print(f"Sampled {len(sampled)} confirmed positives "
      f"(direct={sum(1 for p in sampled if p['surface']=='direct')}, "
      f"indirect={sum(1 for p in sampled if p['surface']=='indirect')})")


def make_score_fn(cat, surface):
    def score_fn(candidate_texts):
        a_batch = mA.score_batch(candidate_texts, [surface]*len(candidate_texts))
        b_batch = mB.score_batch(candidate_texts, [surface]*len(candidate_texts))
        return [max(a.get(cat,0), b.get(cat,0)) for a,b in zip(a_batch, b_batch)]
    return score_fn


def run_attribution(p, beam_width, removal_cap):
    cat  = p["category"]
    thr  = BLOCK_THR[cat]
    fn   = make_score_fn(cat, p["surface"])
    res  = run_removal_beam_search(p["text"], fn, thr, beam_width=beam_width, removal_cap=removal_cap)
    return res


print(f"\nRunning beam_width=5  vs  beam_width=10, removal_cap=6 ...")
print(f"(Score_fn: batched CPU -- TF-IDF+LogReg/MLP; GPU not applicable)\n")

records = []
for idx_s, p in enumerate(sampled):
    res5  = run_attribution(p, beam_width=5,  removal_cap=6)
    res10 = run_attribution(p, beam_width=10, removal_cap=6)
    records.append({
        "sample_idx": p["idx"],
        "surface": p["surface"],
        "attack_type": p["attack_type"],
        "category": p["category"],
        "original_score": round(p["score"],4),
        "beam5":  {"flipped":res5.flipped,  "n_removed":len(res5.removed_tokens),
                   "final_score":round(res5.final_score,4),  "latency_ms":round(res5.search_latency_ms,2)},
        "beam10": {"flipped":res10.flipped, "n_removed":len(res10.removed_tokens),
                   "final_score":round(res10.final_score,4), "latency_ms":round(res10.search_latency_ms,2)},
    })
    if (idx_s+1) % 5 == 0:
        print(f"  {idx_s+1}/{len(sampled)} done ...")

# Aggregate
def agg(key):
    sub = [r for r in records]
    flip5  = [r["beam5"]["flipped"]  for r in sub]
    flip10 = [r["beam10"]["flipped"] for r in sub]
    lat5   = [r["beam5"]["latency_ms"]  for r in sub]
    lat10  = [r["beam10"]["latency_ms"] for r in sub]
    nr5    = [r["beam5"]["n_removed"]  for r in sub if r["beam5"]["flipped"]]
    nr10   = [r["beam10"]["n_removed"] for r in sub if r["beam10"]["flipped"]]
    return {
        "n": len(sub),
        "flip_rate_beam5":  round(float(np.mean(flip5)),3),
        "flip_rate_beam10": round(float(np.mean(flip10)),3),
        "avg_tokens_removed_beam5":  round(float(np.mean(nr5)) if nr5 else 0,2),
        "avg_tokens_removed_beam10": round(float(np.mean(nr10)) if nr10 else 0,2),
        "avg_latency_ms_beam5":  round(float(np.mean(lat5)),2),
        "avg_latency_ms_beam10": round(float(np.mean(lat10)),2),
    }

overall = agg("all")

# Per surface
direct_rec   = [r for r in records if r["surface"]=="direct"]
indirect_rec = [r for r in records if r["surface"]=="indirect"]

def surface_agg(recs, label):
    if not recs: return {}
    f5  = [r["beam5"]["flipped"] for r in recs]
    f10 = [r["beam10"]["flipped"] for r in recs]
    l5  = [r["beam5"]["latency_ms"] for r in recs]
    l10 = [r["beam10"]["latency_ms"] for r in recs]
    nr5 = [r["beam5"]["n_removed"] for r in recs if r["beam5"]["flipped"]]
    nr10= [r["beam10"]["n_removed"] for r in recs if r["beam10"]["flipped"]]
    return {
        "n": len(recs),
        "flip_rate_beam5":  round(float(np.mean(f5)),3),
        "flip_rate_beam10": round(float(np.mean(f10)),3),
        "avg_tokens_removed_beam5":  round(float(np.mean(nr5)) if nr5 else 0,2),
        "avg_tokens_removed_beam10": round(float(np.mean(nr10)) if nr10 else 0,2),
        "avg_latency_ms_beam5":  round(float(np.mean(l5)),2),
        "avg_latency_ms_beam10": round(float(np.mean(l10)),2),
    }

by_surface = {
    "direct":   surface_agg(direct_rec, "direct"),
    "indirect": surface_agg(indirect_rec, "indirect"),
}

# Per attack type
by_atype = {}
for r in records:
    at = r["attack_type"]
    by_atype.setdefault(at,[]).append(r)
by_atype_agg = {at: {"n":len(recs),
                      "flip_rate_b5":round(float(np.mean([r["beam5"]["flipped"] for r in recs])),3),
                      "flip_rate_b10":round(float(np.mean([r["beam10"]["flipped"] for r in recs])),3)}
                for at,recs in by_atype.items()}

print("\n" + "=" * 65)
print("ATTRIBUTION ABLATION RESULTS")
print("=" * 65)
print(f"\nOverall (n={overall['n']})")
print(f"  Flip rate   beam=5:  {overall['flip_rate_beam5']:.3f}")
print(f"  Flip rate   beam=10: {overall['flip_rate_beam10']:.3f}")
print(f"  Avg removed beam=5:  {overall['avg_tokens_removed_beam5']:.2f} tokens")
print(f"  Avg removed beam=10: {overall['avg_tokens_removed_beam10']:.2f} tokens")
print(f"  Avg latency beam=5:  {overall['avg_latency_ms_beam5']:.1f} ms")
print(f"  Avg latency beam=10: {overall['avg_latency_ms_beam10']:.1f} ms")

for surf, d in by_surface.items():
    if not d: continue
    print(f"\n  Surface={surf} (n={d['n']})")
    print(f"    Flip rate  b5={d['flip_rate_beam5']:.3f} / b10={d['flip_rate_beam10']:.3f}")
    print(f"    Avg latency b5={d['avg_latency_ms_beam5']:.1f}ms / b10={d['avg_latency_ms_beam10']:.1f}ms")

print("\nGPU BATCHING NOTE:")
print("  Strategy D score_fn uses TF-IDF+LogReg/MLP -- CPU-only sklearn.")
print("  Batching is already implemented (score_fn accepts List[str] in one call).")
print("  GPU acceleration is NOT applicable for TF-IDF classifiers; no GPU")
print("  numbers reported. If DeBERTa were the scorer, GPU batch inference")
print("  would yield speedup; that path is documented for future work.")

print("\nNOTE: Output described as APPROXIMATE CONTRASTIVE EXPLANATION.")
print("  The removed token set is a sufficient condition for classification flip,")
print("  found by greedy beam search -- NOT a provably minimal set.")
print("  Different seeds or beam widths may find different (equally valid) subsets.")

output = {
    "timestamp": datetime.datetime.now().isoformat(),
    "n_sampled": len(sampled),
    "beam_widths_tested": [5, 10],
    "removal_cap": 6,
    "explanation_type": "approximate_contrastive",
    "gpu_batching": "NOT_APPLICABLE (TF-IDF+sklearn score_fn is CPU-only)",
    "overall": overall,
    "by_surface": by_surface,
    "by_attack_type": by_atype_agg,
    "records": records,
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR,"attribution_eval.json"),"w",encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved -> {RESULTS_DIR}/attribution_eval.json")
print("\nSection 16 DONE")
