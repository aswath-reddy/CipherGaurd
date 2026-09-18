"""
Phase 3 - Section 20: External / Held-Out Benchmark Evaluation.

Supported benchmarks:
  1. WildGuardTest  -- safety benchmark from Allen AI
  2. JailbreakBench -- standardised jailbreak evaluation (NIPS 2024)
  3. HarmBench      -- comprehensive harmfulness benchmark (ICLR 2024)
  4. InjecAgent     -- indirect prompt injection for agents

Environment status (checked at import time):
  - jailbreakbench:  NOT INSTALLED (pip install jailbreakbench)
  - wildguard:       NOT INSTALLED (pip install wildguard)
  - datasets (HF):   NOT INSTALLED (pip install datasets)  <- needed for HarmBench/InjecAgent
  - requests:        available (standard library / pip install requests)

For each benchmark:
  - If loadable: runs CipherGuard Strategy D on a sample and reports metrics
  - If not loadable: prints NOT AVAILABLE with install instructions
    and documents what would be measured

GROUND RULES:
  - External benchmarks are eval-only, never merged into training data
  - Benchmark train/test splits are respected
  - No threshold re-tuning based on external benchmark results
"""

import os, sys, json, datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
BLOCK_THR  = {"Jailbreak":0.45,"Prompt Injection":0.45,"PII Leakage":0.45,
               "Malicious Tools":0.45,"Hate/Toxicity":0.55}

print("=" * 70)
print("CipherGuard Phase 3 -- Section 20: External Benchmark Evaluation")
print("=" * 70)

results_out = {"timestamp": datetime.datetime.now().isoformat(), "benchmarks": {}}


# ── Availability checks ───────────────────────────────────────────────────────
def check_import(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False

HAS_JBB      = check_import("jailbreakbench")
HAS_WILDGUARD= check_import("wildguard")
HAS_DATASETS = check_import("datasets")
HAS_REQUESTS = check_import("requests")

print(f"\nDependency check:")
print(f"  jailbreakbench : {'OK' if HAS_JBB else 'NOT INSTALLED'}")
print(f"  wildguard      : {'OK' if HAS_WILDGUARD else 'NOT INSTALLED'}")
print(f"  datasets (HF)  : {'OK' if HAS_DATASETS else 'NOT INSTALLED'}")
print(f"  requests       : {'OK' if HAS_REQUESTS else 'NOT INSTALLED'}")


# ── Helper: score CipherGuard on a list of (text, label) pairs ───────────────
def score_cipherguard(items, surface="direct", cat_target="Jailbreak"):
    """
    items: list of {"text": str, "label": int (1=harmful/malicious, 0=benign)}
    Returns: {"TP":int, "FP":int, "FN":int, "TN":int, "Precision":float, "Recall":float, "F1":float}
    """
    from src.classification.model_family_a import ModelFamilyA
    from src.classification.model_family_b import ModelFamilyB
    import numpy as np
    mA = ModelFamilyA.load("models")
    mB = ModelFamilyB.load("models")

    texts = [it["text"] for it in items]
    surfs = [surface] * len(texts)
    batch_a = mA.score_batch(texts, surfs)
    batch_b = mB.score_batch(texts, surfs)
    probs = [max(a.get(cat_target,0), b.get(cat_target,0)) for a,b in zip(batch_a, batch_b)]
    thr   = BLOCK_THR.get(cat_target, 0.5)
    preds = [1 if p >= thr else 0 for p in probs]
    labels= [it["label"] for it in items]

    tp = sum(p==1 and l==1 for p,l in zip(preds,labels))
    fp = sum(p==1 and l==0 for p,l in zip(preds,labels))
    fn = sum(p==0 and l==1 for p,l in zip(preds,labels))
    tn = sum(p==0 and l==0 for p,l in zip(preds,labels))
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1   = 2*prec*rec/max(1e-9,prec+rec)
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,
            "Precision":round(prec,3),"Recall":round(rec,3),"F1":round(f1,3),
            "n":len(items),"n_pos":sum(labels)}


# ═════════════════════════════════════════════════════════════════════════════
# 1. JAILBREAKBENCH
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("1. JailbreakBench (Chao et al., NeurIPS 2024)")
print("─" * 70)
if HAS_JBB:
    try:
        import jailbreakbench as jbb
        dataset = jbb.read_dataset()
        # JBB provides prompts + behaviors; we treat each harmful prompt as positive
        items = []
        for behavior in dataset.behaviors[:50]:  # sample first 50
            items.append({"text": behavior.goal, "label": 1})
        metrics = score_cipherguard(items, surface="direct", cat_target="Jailbreak")
        print(f"  Loaded {len(items)} JailbreakBench behaviors (sample)")
        print(f"  Jailbreak detection: P={metrics['Precision']} R={metrics['Recall']} F1={metrics['F1']}")
        results_out["benchmarks"]["jailbreakbench"] = {"status":"RUN", "metrics":metrics}
    except Exception as e:
        print(f"  PARTIAL FAILURE: {e}")
        results_out["benchmarks"]["jailbreakbench"] = {"status":"ERROR", "error":str(e)}
else:
    msg = ("NOT AVAILABLE -- jailbreakbench not installed.\n"
           "  Install: pip install jailbreakbench\n"
           "  What would be measured:\n"
           "    - JBB provides 100 standard jailbreak behaviors with 10 attack methods each\n"
           "    - CipherGuard would be run on each behavior prompt (surface=direct, cat=Jailbreak)\n"
           "    - Metrics: detection rate (recall), false positive rate on benign behaviors\n"
           "    - Comparison: CipherGuard vs JBB's built-in judge scoring\n"
           "    - Respects JBB's eval-only split (no training overlap)")
    print("  " + msg.replace("\n", "\n  "))
    results_out["benchmarks"]["jailbreakbench"] = {"status":"NOT_AVAILABLE",
        "install":"pip install jailbreakbench", "note": msg}


# ═════════════════════════════════════════════════════════════════════════════
# 2. WILDGUARDTEST (Han et al., 2024)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("2. WildGuardTest (Han et al., 2024 -- Allen AI)")
print("─" * 70)
if HAS_WILDGUARD:
    try:
        from wildguard import load_wildguard_data
        data = load_wildguard_data(split="test")
        items = [{"text": d["instruction"], "label": 1 if d["label"]=="harmful" else 0}
                 for d in data[:100]]
        metrics = score_cipherguard(items, surface="direct", cat_target="Jailbreak")
        print(f"  WildGuardTest loaded: {len(items)} samples")
        print(f"  Detection: P={metrics['Precision']} R={metrics['Recall']} F1={metrics['F1']}")
        results_out["benchmarks"]["wildguardtest"] = {"status":"RUN","metrics":metrics}
    except Exception as e:
        print(f"  PARTIAL FAILURE: {e}")
        results_out["benchmarks"]["wildguardtest"] = {"status":"ERROR","error":str(e)}
else:
    msg = ("NOT AVAILABLE -- wildguard package not installed.\n"
           "  Install: pip install wildguard  (or follow repo README: allenai/wildguard)\n"
           "  What would be measured:\n"
           "    - WildGuardTest has 1725 test prompts (harmful/benign/refusal)\n"
           "    - CipherGuard would be evaluated on the harmful vs benign split\n"
           "    - Metrics: precision, recall, F1 on multi-category labels\n"
           "    - Key comparison: WildGuard model accuracy vs CipherGuard on same inputs")
    print("  " + msg.replace("\n","\n  "))
    results_out["benchmarks"]["wildguardtest"] = {"status":"NOT_AVAILABLE",
        "install":"pip install wildguard", "note": msg}


# ═════════════════════════════════════════════════════════════════════════════
# 3. HARMBENCH (Mazeika et al., ICLR 2024)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("3. HarmBench (Mazeika et al., ICLR 2024)")
print("─" * 70)
if HAS_DATASETS:
    try:
        from datasets import load_dataset
        # HarmBench is hosted as a HuggingFace dataset
        ds = load_dataset("walledai/HarmBench", split="test")
        items = [{"text": r["prompt"], "label": 1} for r in list(ds)[:100]]
        metrics = score_cipherguard(items, surface="direct", cat_target="Jailbreak")
        print(f"  HarmBench loaded: {len(items)} samples (eval-only)")
        print(f"  Detection: P={metrics['Precision']} R={metrics['Recall']} F1={metrics['F1']}")
        results_out["benchmarks"]["harmbench"] = {"status":"RUN","metrics":metrics}
    except Exception as e:
        print(f"  PARTIAL FAILURE: {e}")
        results_out["benchmarks"]["harmbench"] = {"status":"ERROR","error":str(e)}
else:
    msg = ("NOT AVAILABLE -- HuggingFace datasets not installed.\n"
           "  Install: pip install datasets\n"
           "  Dataset: walledai/HarmBench on HuggingFace Hub\n"
           "  What would be measured:\n"
           "    - HarmBench provides 400 harmful behaviors across 7 functional categories\n"
           "    - CipherGuard would be run on the standard_behaviors test split\n"
           "    - Metrics: Attack Success Rate detected (recall), per-category breakdown\n"
           "    - Note: HarmBench evaluates attack success, not just prompt risk;\n"
           "      CipherGuard evaluates prompt risk at gateway -- complementary, not identical")
    print("  " + msg.replace("\n","\n  "))
    results_out["benchmarks"]["harmbench"] = {"status":"NOT_AVAILABLE",
        "install":"pip install datasets", "note": msg}


# ═════════════════════════════════════════════════════════════════════════════
# 4. INJECAGENT (Zhan et al., 2024) -- Indirect Prompt Injection for Agents
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("4. InjecAgent (Zhan et al., 2024 -- indirect injection benchmark)")
print("─" * 70)
if HAS_DATASETS:
    try:
        from datasets import load_dataset
        ds = load_dataset("abcdabcd987/InjecAgent", split="test")
        # InjecAgent has injected_prompt field in tool outputs
        items = [{"text": r.get("injected_prompt", r.get("user_task","")), "label": 1}
                 for r in list(ds)[:50]]
        metrics = score_cipherguard(items, surface="indirect", cat_target="Prompt Injection")
        print(f"  InjecAgent loaded: {len(items)} samples (indirect surface)")
        print(f"  PI detection: P={metrics['Precision']} R={metrics['Recall']} F1={metrics['F1']}")
        results_out["benchmarks"]["injecagent"] = {"status":"RUN","metrics":metrics}
    except Exception as e:
        print(f"  PARTIAL FAILURE: {e}")
        results_out["benchmarks"]["injecagent"] = {"status":"ERROR","error":str(e)}
else:
    msg = ("NOT AVAILABLE -- HuggingFace datasets not installed.\n"
           "  Install: pip install datasets\n"
           "  Dataset: abcdabcd987/InjecAgent on HuggingFace Hub (or local clone)\n"
           "  What would be measured:\n"
           "    - InjecAgent provides 1054 test cases for indirect injection in agent tasks\n"
           "    - Content is injected into tool outputs (web search, email, calendar, etc.)\n"
           "    - CipherGuard would run on the injected_prompt field with surface=indirect\n"
           "    - Category: Prompt Injection (task-hijacking)\n"
           "    - This is the most relevant external benchmark for CipherGuard's indirect surface\n"
           "    - Metrics: detection rate, FP rate on non-injected tool outputs\n"
           "    - PRIORITY: install this benchmark first for indirect injection validation")
    print("  " + msg.replace("\n","\n  "))
    results_out["benchmarks"]["injecagent"] = {"status":"NOT_AVAILABLE",
        "install":"pip install datasets", "priority":"HIGH", "note": msg}


# ═════════════════════════════════════════════════════════════════════════════
# 5. AGENTDOJO (Debenedetti et al., NeurIPS 2024)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("5. AgentDojo (Debenedetti et al., NeurIPS 2024)")
print("─" * 70)
HAS_AGENTDOJO = check_import("agentdojo")
if HAS_AGENTDOJO:
    print("  agentdojo found -- integration NOT implemented yet")
    results_out["benchmarks"]["agentdojo"] = {"status":"FOUND_NOT_INTEGRATED"}
else:
    msg = ("NOT AVAILABLE -- agentdojo not installed.\n"
           "  Install: pip install agentdojo\n"
           "  What would be measured:\n"
           "    - AgentDojo provides 97 injection tasks across 5 agent environments\n"
           "    - It measures end-to-end attack success vs task utility\n"
           "    - CipherGuard would be inserted as a gateway before tool-output processing\n"
           "    - Requires running a full agent loop -- not just prompt classification\n"
           "    - Integration complexity: HIGH (not a drop-in classifier benchmark)\n"
           "    - Suitable for future integration testing, not Phase 3 evaluation")
    print("  " + msg.replace("\n","\n  "))
    results_out["benchmarks"]["agentdojo"] = {"status":"NOT_AVAILABLE",
        "install":"pip install agentdojo", "note": msg}


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXTERNAL BENCHMARK SUMMARY")
print("=" * 70)
for name, d in results_out["benchmarks"].items():
    status = d["status"]
    if status == "RUN":
        m = d.get("metrics",{})
        print(f"  {name:<20}: RUN    F1={m.get('F1','?')} P={m.get('Precision','?')} R={m.get('Recall','?')}")
    elif status == "ERROR":
        print(f"  {name:<20}: ERROR  {d.get('error','?')[:60]}")
    else:
        priority = " [HIGH PRIORITY]" if d.get("priority") == "HIGH" else ""
        print(f"  {name:<20}: NOT_AVAILABLE{priority}  -- {d.get('install','')}")

with open(os.path.join(RESULTS_DIR,"external_benchmark_results.json"),"w",encoding="utf-8") as f:
    json.dump(results_out, f, indent=2)
print(f"\nSaved -> {RESULTS_DIR}/external_benchmark_results.json")
print("\nSection 20 DONE")
