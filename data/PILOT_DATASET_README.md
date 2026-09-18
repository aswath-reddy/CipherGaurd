# Pilot Dataset — n=85 Proof-of-Concept

## Role in the Project

`pilot_dataset_85.json` is the **Pilot / Proof-of-Concept dataset** for CipherGuard.
It was hand-curated with 85 samples to validate the core pipeline before larger-scale
data collection and is used primarily for:

- Sanity-checking the end-to-end classification pipeline
- Initial feasibility demonstrations and presentations
- Regression testing (quick smoke test that the API returns expected categories)

## What It Is NOT

| NOT for | Reason |
|---|---|
| Model training | 85 samples is far too small for reliable generalisation |
| Final evaluation / test set | Would conflate development and evaluation |
| Hyperparameter tuning | Would cause data leakage |
| Reporting final accuracy numbers | Results would not be statistically meaningful |

The project uses a separate, larger dataset (`expanded_dataset.json` / the three
split files `train.json`, `val.json`, `test.json`) for all training and evaluation.

## Schema

Binary-label format (legacy). Each record:

```json
{
  "id": 1,
  "surface": "direct | indirect",
  "label": 0,
  "text": "..."
}
```

`label = 0` → benign  
`label = 1` → malicious (Jailbreak / Prompt Injection for direct surface;
Prompt Injection for indirect surface)

## Statistics

| Metric | Value |
|---|---|
| Total samples | 85 |
| Direct benign | 30 |
| Direct malicious | 25 |
| Indirect benign | 15 |
| Indirect malicious | 15 |

## Integrity

The file must **never be modified**. A SHA-256 checksum is enforced by
`evaluation/build_dataset.py` — any modification will cause the pipeline to abort
with a clear error message.

**SHA-256 (authoritative):** computed on first run and stored in
`data/.pilot_checksum` (generated automatically; do not commit this file if the
dataset file is tracked separately).

## Citation / Credit

Hand-curated by the CipherGuard project team for final-year academic submission,
2025–2026. Samples cover direct and indirect attack surfaces across Jailbreak and
Prompt Injection categories.
