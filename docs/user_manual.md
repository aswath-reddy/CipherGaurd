# CipherGuard User Manual & Operations Guide

This guide describes how to run, configure, and operate the CipherGuard safety routing gateway.

---

## 1. Installation & Environment Setup

Ensure Python 3.10+ is installed on your system. From the project root (`C:\Users\rpich\OneDrive\Desktop\P\cipherguard`):

```bash
# Install core and dev dependencies
pip install -e .
pip install streamlit pytest
```

---

## 2. Launching the Interactive Streamlit Demo UI

The Streamlit UI provides a visual dashboard to test inputs across surfaces, inspect risk scores, view contrastive explanations, and hot-reload policies.

```bash
streamlit run ui/app.py
```

### Using the UI:
1. **Surface Selection**: Select **Direct Attack Surface** for user-turn prompts or **Indirect Injection Surface** for retrieved documents / tool outputs.
2. **Presets**: Choose from pre-configured adversarial and safe examples or enter custom text.
3. **Inspect & Route**: Click **Inspect & Route Input**. If the input breaches a policy threshold:
   - A **BLOCK** or **REVIEW** badge appears.
   - The **Contrastive Token Attribution Panel** highlights the exact adversarial tokens identified for removal.
   - The **Sanitized Content** preview displays the clean text with malicious tokens excised.
4. **Dynamic Policy Tab**: Modify threshold values or actions in the JSON editor and click **Save & Hot-Reload Policy**.

---

## 3. Running the REST API Gateway

To run the production FastAPI service:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger / OpenAPI documentation will be available at:
`http://localhost:8000/docs`

---

## 4. Hot-Reloading Safety Policies

CipherGuard allows administrators to modify safety policies at runtime with zero classifier downtime.

1. Open `policy/policy.json` in any text editor.
2. Edit a threshold (e.g., lower the indirect threshold from `0.42` to `0.30`).
3. Save the file.
4. CipherGuard detects the file modification and re-validates the schema automatically on the next request.
5. If an invalid JSON syntax error is introduced, CipherGuard's fail-safe mechanism safely retains the previous valid policy.

---

## 5. Running Tests and Evaluations

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Repeated Cross-Validation (Table I)
```bash
python -m evaluation.run_cv
```

### Run Attribution Budget Ablation
```bash
python -m evaluation.run_budget_ablation
```

### Run Zero-Shot Semantic Evaluation (DeBERTa-v3, Table II)
```bash
python -m evaluation.run_zero_shot_compare
```
