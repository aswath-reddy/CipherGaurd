# CipherGuard Empirical Evaluation Report

This report presents empirical results evaluating CipherGuard's three core mechanisms:
1. **Dynamic Policy Decoupling**
2. **Dual-Surface Threat Calibration**
3. **Removal-Based Contrastive Token Attribution**

Evaluations were executed across both lightweight discriminators (TF-IDF + Logistic Regression and MLP) and a production-grade transformer semantic classifier (`protectai/deberta-v3-base-prompt-injection-v2`) on the pilot dataset ($n=85$: 45 safe, 40 unsafe; 55 direct, 30 indirect).

---

## 1. Classification & Model Family Evaluation (Table I Reproduction)

We evaluated Model Family A (Linear Discriminator: TF-IDF + Logistic Regression) and Model Family B (Nonlinear Discriminator: TF-IDF + 32-unit MLP) using **repeated stratified 5-fold cross-validation repeated 5 times (25 folds total)**.

### Cross-Validation Results ($n=85$, 25 folds)

| Metric | Model Family A (LogReg) | Model Family B (MLP) |
| :--- | :--- | :--- |
| **Accuracy** | **0.962** ($\pm 0.037$) | **0.899** ($\pm 0.074$) |
| **Precision** | **0.959** ($\pm 0.055$) | **0.857** ($\pm 0.113$) |
| **Recall** | **0.965** ($\pm 0.056$) | **0.970** ($\pm 0.064$) |
| **F1-Score** | **0.960** ($\pm 0.039$) | **0.904** ($\pm 0.064$) |

### Findings
* Both classifiers demonstrate strong separability on the synthetic pilot dataset.
* Model Family A achieves higher precision ($0.959$), whereas Model Family B exhibits marginally higher recall ($0.970$), confirming that model family selection inherently trades off operating points even when trained on identical input representations.

---

## 2. Attribution Search Budget Ablation (Table I Reproduction)

To isolate search-budget constraints from model capacity limitations, we evaluated the removal-based beam search under two distinct search budgets across 40 blocked instances:
1. **Narrow Budget**: Beam width $b=5$, removal cap $k \le 6$
2. **Wide Budget**: Beam width $b=10$, removal cap $k \le 10$

### Budget Ablation Results

| Budget Configuration | Flip Rate (% Passed) | Mean Tokens Removed ($|\Delta S|$) | Mean Search Latency |
| :--- | :--- | :--- | :--- |
| **Narrow ($b=5, k \le 6$)** | 0.0% (0/40) | — | **20.0 ms** |
| **Wide ($b=10, k \le 10$)** | **12.5%** (5/40) | 9.20 tokens | **37.6 ms** |

### Findings
* Widening the search budget from $(5, 6)$ to $(10, 10)$ enabled flips that were inaccessible under the narrow budget.
* However, successful flips required removing an average of 9.2 tokens (close to the cap), indicating that in bag-of-words linear models, discriminative signal is diffused across multiple tokens rather than concentrated in a compact phrase.

---

## 3. Real Semantic Classifier Benchmark (Table II Reproduction)

To benchmark against deep contextual encoders, we evaluated `protectai/deberta-v3-base-prompt-injection-v2` zero-shot on the identical $n=85$ dataset.

### Classification & Surface Asymmetry Results

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **0.977** |
| **Overall Precision** | **1.000** (Zero False Positives) |
| **Overall Recall** | **0.950** |
| **Direct Surface Recall** ($S_{\text{dir}}$) | **0.920** |
| **Indirect Surface Recall** ($S_{\text{ind}}$) | **1.000** |

### Semantic Attribution Performance
* **Flip Rate (Narrow budget: $b=5, k \le 6$)**: **84.2%** (32 / 38 blocked instances successfully flipped).
* **Mean Tokens Removed**: **2.16 tokens** (compact, highly concise counterfactuals).
* **Mean Search Latency**: 6.01 seconds (evaluated in batched mode on CPU).

### Findings
1. **Surface Asymmetry Confirmation**: DeBERTa achieved 100% recall on the indirect surface, compared to 92% on direct jailbreaks. This confirms the paper's core hypothesis: surface difficulty is an artifact of classifier training distribution, necessitating decoupled per-surface calibration.
2. **Superior Explanation Minimality**: Unlike TF-IDF models where signal is diffused, the transformer's attention concentrates heavily on recognizable injection triggers. As a result, removing an average of just **2.16 tokens** (e.g., `"ignore previous instructions"`) was sufficient to flip the decision to PASS.

---

## 4. Disclosed Limitations & Negative Results

Following the scientific honesty standard of the research paper:
1. **CPU Latency Ceiling**: In CPU environments without CUDA acceleration, transformer beam search averages $\approx 6$ seconds per explanation. For real-time production deployment, GPU acceleration or ONNX Runtime quantization is required.
2. **Synthetic Dataset Scale**: The pilot evaluation operates on $n=85$ synthetic examples. As reported in the paper, naive scaling via slot-filling templates causes artificial separability (accuracy $\approx 0.998$); future work must incorporate diverse human-red-teamed corpora (e.g., InjecAgent and HarmBench).
