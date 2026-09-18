# CipherGuard — Model Architecture Decisions

## DeBERTa Fine-Tuned Model: Archived

### What exists

`models/cipherguard-deberta-finetuned-custom/` contains a LoRA-adapted
`DebertaV2ForSequenceClassification` checkpoint trained on the pre-Phase-1
CipherGuard dataset (~704 samples) for 88 gradient steps (checkpoint-88).

### Why it is archived (not activated)

| Reason | Detail |
|---|---|
| **Stale training data** | Trained before Phase 1 expanded the dataset. The training corpus had zero PII Leakage positives and no indirect-surface samples. The expanded dataset (Phase 1) is structurally different. |
| **Insufficient training** | 88 gradient steps on a 704-sample dataset is far below the minimum for reliable multi-label fine-tuning of a DeBERTa-v3-base model (~86M parameters). Typical practice requires at minimum 3–5 full epochs (≥2,000 steps for this dataset size). |
| **Label mapping not verified** | `config.json` maps outputs to `LABEL_0`–`LABEL_4` with no category names. The ordering is unconfirmed relative to CipherGuard's `[Jailbreak, Prompt Injection, PII Leakage, Malicious Tools, Hate/Toxicity]` ordering used in Phase 1+. Applying mis-ordered labels would silently corrupt outputs. |
| **False positive behaviour** | Initial testing showed the fine-tuned checkpoint produced false positives on benign inputs — consistent with a model undertrained on an imbalanced corpus (prevalence 7–17% per category). |
| **LoRA not merged** | `adapter_model.safetensors` (2.3 MB) is a LoRA delta only. The base model weights are stored separately. The adapter has not been merged into the base, adding deployment complexity with no quality benefit at current scale. |

### Why the pretrained ProtectAI checkpoint is currently preferred

`protectai/deberta-v3-base-prompt-injection-v2` is used in zero-shot mode via
`FinetunedDebertaClassifier` (which falls back automatically when no fine-tuned
checkpoint is found at `models/cipherguard-deberta-finetuned/`).

| Property | Value |
|---|---|
| Model | ProtectAI DeBERTa-v3-base fine-tuned on prompt-injection data |
| Categories covered | Jailbreak, Prompt Injection (binary output: SAFE / INJECTION) |
| Training data | Large, curated, public prompt-injection dataset (not our data) |
| Reported accuracy | 97.7% on CipherGuard pilot benchmark (§3, evaluation_report.md) |
| False positive risk | Low — purpose-built for the Jailbreak/PI detection task |
| Deployment | Zero-shot, no training required, weights cached locally |

### Code structure for future swap-in

`FinetunedDebertaClassifier` in `src/classification/finetuned_deberta_classifier.py`
is already structured for clean hot-swap:

```python
FINETUNED_MODEL_DIR = "models/cipherguard-deberta-finetuned"  # change this path
```

When a new fine-tuned checkpoint is saved to that directory with a valid `config.json`
and proper `id2label` mapping, the system will automatically load it on next startup
without any code changes. The fallback to zero-shot base remains in place if loading fails.

### What would justify revisiting fine-tuning

| Criterion | Current state | Required for fine-tuning |
|---|---|---|
| Training samples per category | 47–111 positives (train split) | ≥1,000 positives per category |
| Total dataset size | 923 samples | ≥5,000 samples |
| GPU available | CUDA available (torch 2.6.0+cu124) | ≥16 GB VRAM for full fine-tune; 8 GB for LoRA |
| Label config verified | No (LABEL_0–4 unordered) | Must set `id2label` before training |
| Validated improvement | Not yet tested rigorously | Must show val F1 gain ≥5pp over ProtectAI baseline |
| Indirect surface representation | 10.2% of dataset | ≥20% recommended for surface-aware fine-tuning |
| Human red-team coverage | Minimal | Include InjecAgent / HarmBench samples |

---

## Ensemble Fusion Decision

See `evaluation/results/best_strategy.json` and `evaluation/results/ablation_val_results.json`
for the experimental evidence behind the selected fusion strategy.

The current production strategy is documented in `src/classification/risk_aggregator.py`.
The winning strategy from the Phase 2 ablation (A–G) is recorded in `best_strategy.json`
and applied in `evaluation/phase2_final_test_eval.py`.

---

## Threshold Decision

See `evaluation/results/thresholds_frozen.json`.

Thresholds were selected by minimising `0.7 * FNR + 0.3 * FPR` on the validation
split (138 samples, seed=42). FNR is weighted 2.3× higher than FPR to reflect
the security context: missing an attack is more costly than a false alarm.

Separate per-surface thresholds for Jailbreak and Prompt Injection are reported
where the indirect validation subsplit (n=14) contained sufficient positives.
These per-surface thresholds should be treated as indicative only given the
small indirect-surface sample count.
