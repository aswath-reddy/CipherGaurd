# Software Requirements Specification (SRS)
## CipherGuard: A Policy-Aware Safety Router with Removal-Based Token Attribution for LLM Applications

**Institution:** Keshav Memorial Institute of Technology (KMIT), Hyderabad  
**Department:** Computer Science & Engineering  
**Project Guide:** B. Shailesh  
**Project Team:** Shreya Namdeo, Shashank Reddy Yasa, Adi Aswatha Reddy, B. Abhinav  
**Version:** 1.0.0  
**Date:** September 2026  

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) establishes the functional, performance, security, and interface requirements for **CipherGuard**, an auditable, pre-agent safety gateway for Large Language Model (LLM) applications. It serves as the primary technical specification for the project review, development, testing, and academic verification.

### 1.2 Scope
CipherGuard sits directly between untrusted inputs and core LLM agents. It decouples safety policy management from classifier weights and provides contrastive token-level attribution for blocked or flagged content. Specifically, CipherGuard:
- Ingests direct user prompts and indirect third-party content (retrieved documents, tool outputs, emails, API payloads).
- Classifies multi-dimensional safety risks (Jailbreak, Prompt Injection, PII Leakage, Malicious Tool Use, Hate/Toxicity).
- Evaluates risks dynamically against a JSON-configured policy matrix without requiring model retraining.
- Applies a removal-based beam search to identify the minimal token subset causing a violation.
- Produces contrastive explanations, risk delta scores ($\Delta s$), and sanitized safe variants.
- Persists full audit trails into an auditable logging store.

### 1.3 Definitions, Acronyms, and Abbreviations
- **LLM:** Large Language Model.
- **IPI:** Indirect Prompt Injection (attacks delivered via external retrieved context).
- **DPI:** Direct Prompt Injection (jailbreak attempts directly entered by the user).
- **Dynamic Policy Matrix:** A declarative runtime JSON rule set that maps threat categories and input surfaces to thresholds and routing actions.
- **Contrastive Token Attribution:** An explainability method identifying the minimal subset of words whose removal flips a decision from `BLOCK` to `ALLOW`.
- **RBAC:** Role-Based Access Control.
- **SRS:** Software Requirements Specification.
- **API:** Application Programming Interface (REST).

### 1.4 References
1. Inan et al., *Llama Guard: LLM-Based Input-Output Safeguard for Human-AI Conversations*, Meta AI, arXiv:2312.06674, 2023.
2. Zhan et al., *InjecAgent: Benchmarking Indirect Prompt Injections*, ACL Findings 2024.
3. Debenedetti et al., *AgentDojo: Evaluating Prompt Injection Attacks*, NeurIPS D&B 2024.
4. Protect AI, *deberta-v3-base-prompt-injection-v2*, Hugging Face, 2024.
5. Rebedea et al., *NeMo Guardrails*, EMNLP System Demonstrations, 2023.

---

## 2. Overall Description

### 2.1 Product Perspective
CipherGuard is a self-contained, pre-agent security gateway. Rather than relying solely on monolithic, static models (such as Llama Guard) or high-latency LLM-as-a-judge evaluators, CipherGuard operates as a lightweight inline proxy. 

```
[ Direct User Prompt ]       ──────┐
                                   ▼
                            ┌───────────────┐
                            │  CipherGuard  │ ──(ALLOW)──► [ Core LLM Agent ]
                            │ Safety Router │
                            └───────┬───────┘
                                    │ (BLOCK / REVIEW)
[ Indirect Tool / Docs ]    ───────┘▼
                            ┌───────────────────────────────┐
                            │ Contrastive Explanation Engine│
                            │ & SQLite Audit Log Store      │
                            └───────────────────────────────┘
```

### 2.2 Product Functions
1. **Dual-Surface Ingestion:** Normalizes direct user queries and indirect context payloads while tracking surface provenance.
2. **Multi-Model Risk Scoring:** Computes risk probabilities across categories using ensemble and semantic models.
3. **Dynamic Rule Resolution:** Evaluates risk vectors against hot-reloadable declarative JSON policies with hierarchical fallback (`surface-specific` $\rightarrow$ `wildcard *`).
4. **Removal-Based Beam Search:** Computes minimal token removal combinations flipping `BLOCK` $\rightarrow$ `PASS`.
5. **Sanitization & Redaction:** Reconstructs a clean version of the input stripping malicious trigger tokens.
6. **Audit Trail Persistence:** Records input text, risk vectors, rule evaluations, attribution statistics, and latency metrics.

### 2.3 User Classes and Characteristics
- **Security Administrator / Compliance Auditor:** Configures policies, reviews audit logs, inspects attribution rationale, adjusts risk thresholds.
- **AI Application Developer:** Integrates CipherGuard REST endpoints or Python SDK into agentic workflows and LangChain/LlamaIndex pipelines.
- **End User:** Interacts with the LLM application; receives clear contrastive explanations when requests are rejected rather than opaque generic errors.

### 2.4 Operating Environment
- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), macOS.
- **Runtime:** Python 3.10, 3.11, or 3.12.
- **Frameworks:** FastAPI, Uvicorn, Streamlit, PyTorch / Hugging Face Transformers, scikit-learn, SQLite3.

### 2.5 Design Constraints
- Real-time routing decisions for benign prompts must resolve with sub-50ms latency when using lightweight models.
- Policy matrix updates must apply instantaneously without restarting server processes or reloading model weights into memory.
- The system must function entirely offline/on-premise without external cloud API dependencies.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### Module 1: Ingestion & Surface Normalization
- **FR-1.1 (Direct Input):** The system shall accept user text prompts marked with surface identifier `direct`.
- **FR-1.2 (Indirect Input):** The system shall accept external content (documents, emails, retrieved chunks, tool outputs) marked with surface identifier `indirect` and metadata source types.
- **FR-1.3 (Provenance Preservation):** The system shall retain input metadata and source context throughout the routing pipeline.

#### Module 2: Classification & Risk Aggregation
- **FR-2.1 (Multi-Family Classifier):** The system shall support ensembling across Model Family A (lexical / n-gram TF-IDF + Logistic Regression), Model Family B (dense embeddings + MLP), and optional fine-tuned DeBERTa-v3 semantic classifiers.
- **FR-2.2 (Multi-Dimensional Risk Vector):** The classifier shall output calibrated scores in $[0.0, 1.0]$ for categories including `Jailbreak`, `Prompt Injection`, `PII Leakage`, `Malicious Tools`, and `Hate/Toxicity`.
- **FR-2.3 (Batch Inference):** The classification layer shall provide batch scoring interfaces to support parallel attribution search evaluations.

#### Module 3: Dynamic Policy Engine
- **FR-3.1 (JSON Policy Schema):** Policies shall be defined in a human-readable JSON schema specifying category, surface (`direct`, `indirect`, or `*`), numeric threshold, and routing action (`BLOCK`, `REVIEW`, `ALLOW`).
- **FR-3.2 (Hot Reloading):** The policy loader shall detect configuration file changes on disk or via API and reload policies at runtime without service disruption.
- **FR-3.3 (Fallback & Validation):** If an invalid policy file is supplied, the system shall reject the change, log a warning, and fall back to the last known valid configuration.

#### Module 4: Removal-Based Token Attribution
- **FR-4.1 (Trigger Condition):** Removal-based attribution shall automatically execute whenever an input yields a `BLOCK` or `REVIEW` action.
- **FR-4.2 (Beam Search Optimization):** The engine shall use a configurable beam width ($k \in [1, 10]$) and maximum removal cap ($L \in [1, 10]$) to find the minimal token removal set that reduces risk score below the category threshold.
- **FR-4.3 (Contrastive Output):** The attribution engine shall output:
  - List of removed tokens with positional indices.
  - Cardinality of removal set ($k^*$).
  - Score reduction ($\Delta s = s_{\text{orig}} - s_{\text{sanitized}}$).
  - Decision flip verification (`is_flipped` boolean flag).
  - Sanitized text variant with offending tokens excised.

#### Module 5: Audit & Administration
- **FR-5.1 (Persistent Audit Logging):** Every routing transaction shall be recorded in an SQLite database with timestamp, input text, risk vector, action, rationale, attribution summary, and latency.
- **FR-5.2 (REST API):** The system shall expose endpoints:
  - `POST /route`: End-to-end evaluation.
  - `GET /policy` & `PUT /policy`: Retrieve and hot-update policy configurations.
  - `GET /logs`: Query historical audit records.
  - `POST /explain`: Standalone attribution computation on arbitrary text.
- **FR-5.3 (Interactive UI):** A Streamlit dashboard shall provide real-time input testing, interactive policy editing, and visual attribution inspection.

---

### 3.2 Non-Functional Requirements

#### NFR-1: Performance & Latency
- Benign request classification through lightweight classifiers must complete in $\le 50\text{ ms}$.
- Removal attribution beam search must complete in $\le 400\text{ ms}$ for typical input sequences ($\le 60$ tokens) with beam width $k=5$.

#### NFR-2: Security & Integrity
- All policy updates submitted via REST API must be validated against a strict Pydantic schema before acceptance.
- SQL queries to the audit store must use parameterized statements to prevent SQL injection.
- The router must fail closed (`BLOCK`) if an internal classification exception occurs.

#### NFR-3: Reliability & Availability
- The system shall maintain 99.9% uptime during runtime policy swaps.
- SQLite database transactions must use write-ahead logging (WAL) or synchronous integrity to prevent database corruption.

#### NFR-4: Maintainability & Extensibility
- The classification interface must use an abstract base class (`BaseClassifier`) enabling plug-and-play addition of new classifiers (e.g., Llama Guard, ShieldGemma).
- New safety categories can be added via policy JSON without altering the routing engine source code.

#### NFR-5: Usability & Transparency
- Rejection responses must clearly articulate which rule fired, the threshold breach, and the exact words triggering the block.

---

## 4. Verification & Acceptance Criteria
1. **Automated Testing:** 100% pass rate across unit and integration tests covering router logic, policy parsing, beam search attribution, and classifier scoring.
2. **Hot-Reload Verification:** Modifying `policy.json` thresholds while sending requests immediately alters routing behavior without server restart.
3. **Attribution Correctness:** Verified test prompts containing known jailbreak triggers (e.g., "Ignore all previous instructions...") must accurately identify trigger words as minimal removal set.
