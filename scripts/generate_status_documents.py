"""
Generates professional DOCX and PDF versions of the CipherGuard Project Status Report
(Inception to Phase 1) without any personal, institutional, or member details.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.pdfgen import canvas

# -----------------------------------------------------------------------------
# Color Palette Constants
# -----------------------------------------------------------------------------
HEX_PRIMARY = "0F2942"      # Deep Navy
HEX_SECONDARY = "1E5F8A"    # Steel Blue
HEX_ACCENT = "0D9488"       # Teal
HEX_DARK = "1E293B"         # Charcoal Text
HEX_LIGHT_BG = "F1F5F9"     # Light Slate Background
HEX_BORDER = "CBD5E1"       # Border Grey

COLOR_PRIMARY = RGBColor(15, 41, 66)
COLOR_SECONDARY = RGBColor(30, 95, 138)
COLOR_DARK = RGBColor(30, 41, 59)
COLOR_MUTED = RGBColor(100, 116, 139)

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def generate_docx(filename):
    doc = Document()
    
    # Page Margins (0.75 in)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
        # Add page numbering to footer
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("CipherGuard Technical Report | Confidential")
        f_run.font.size = Pt(8.5)
        f_run.font.color.rgb = COLOR_MUTED

    # Header Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("CipherGuard: Project Status Report")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(12)
    run_sub = sub_p.add_run("System Architecture, Inception to Phase 1 Milestone, Findings, and Next Phases Roadmap")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = COLOR_SECONDARY

    # Metadata Strip Box
    meta_table = doc.add_table(rows=1, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    col_widths = [Inches(2.3), Inches(2.3), Inches(2.4)]
    meta_data = [
        ("Current Milestone", "Phase 1 Complete (Dataset & Baseline)"),
        ("System Status", "Live REST API & UI Active"),
        ("Test Verification", "13 / 13 Automated Tests Passing (100%)")
    ]
    for i, (k, v) in enumerate(meta_data):
        cell = meta_table.rows[0].cells[i]
        cell.width = col_widths[i]
        set_cell_background(cell, HEX_LIGHT_BG)
        set_cell_margins(cell, 120, 120, 140, 140)
        cp = cell.paragraphs[0]
        cp.paragraph_format.space_after = Pt(2)
        r_k = cp.add_run(k.upper() + "\n")
        r_k.font.size = Pt(8)
        r_k.font.bold = True
        r_k.font.color.rgb = COLOR_MUTED
        r_v = cp.add_run(v)
        r_v.font.size = Pt(9.5)
        r_v.font.bold = True
        r_v.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    def add_section_header(text, level=1):
        p = doc.add_paragraph()
        if level == 1:
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = COLOR_PRIMARY
        elif level == 2:
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = COLOR_SECONDARY

    def add_body(text, bold_prefix=None, space_after=4):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_DARK
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = COLOR_DARK

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_DARK
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = COLOR_DARK

    # 1. Executive Summary
    add_section_header("1. Executive Summary & Core Motivation", 1)
    add_body(
        "Modern Large Language Model (LLM) agent workflows face severe security vulnerabilities from direct user "
        "jailbreaks (DPI) and indirect prompt injections (IPI) embedded in external retrieved context (RAG), database "
        "records, tool outputs, and incoming emails. Existing safety solutions suffer from two fundamental bottlenecks:"
    )
    add_bullet(
        "Hardcodes moderation policies directly into model parameters. Modifying a risk threshold or "
        "introducing an enterprise-specific boundary requires computationally expensive model retraining or fine-tuning.",
        bold_prefix="Monolithic Classifier Rigidity (e.g., Llama Guard, ShieldGemma): "
    )
    add_bullet(
        "Invoking secondary generative LLMs to evaluate input safety incurs high per-token API costs, adds "
        "hundreds of milliseconds or seconds of latency, and yields non-deterministic, unauditable decisions.",
        bold_prefix="LLM-as-a-Judge Overhead: "
    )
    add_body(
        "CipherGuard bridges this gap by providing an inline, auditable pre-agent safety gateway characterized by "
        "three primary technical innovations:"
    )
    add_bullet(
        "A hot-reloadable, declarative JSON policy matrix that decouples runtime routing decisions "
        "(BLOCK, REVIEW, ALLOW) from classifier weights, enabling zero-downtime threshold adjustments.",
        bold_prefix="Dynamic Policy Decoupling: "
    )
    add_bullet(
        "Independent risk scoring and threshold calibration tailored to the distinct operational "
        "characteristics of user turns ('direct') versus ingested third-party data ('indirect').",
        bold_prefix="Dual-Surface Threat Modeling: "
    )
    add_bullet(
        "An approximate beam search engine that isolates the minimal subset of trigger tokens whose "
        "removal flips a rejection to a pass, generating both verifiable audit rationale and safe sanitized text.",
        bold_prefix="Removal-Based Contrastive Attribution: "
    )

    # 2. System Architecture
    add_section_header("2. System Architecture & Subsystems", 1)
    add_body(
        "CipherGuard is structured as an inline pipeline that intercepts untrusted payloads before they reach core LLMs. "
        "The architecture is partitioned into five decoupled subsystems:"
    )
    add_bullet(
        "Accepts user prompts (DirectInput) and external payloads (IndirectInput) including retrieved documents, "
        "tool responses, emails, and web text while preserving surface provenance throughout routing.",
        bold_prefix="1. Ingestion & Normalization: "
    )
    add_bullet(
        "Multi-model ensemble comprising linear discriminators (ModelFamilyA: TF-IDF + LogReg), non-linear neural nets "
        "(ModelFamilyB: TF-IDF + MLP), deep semantic transformers (DeBERTa-v3), hate/toxicity models (ToxicBERT), and PII analyzers (Presidio).",
        bold_prefix="2. Classification & Risk Fusion: "
    )
    add_bullet(
        "A JSON-configured rules engine (policy/policy.json) with strict Pydantic validation, wildcard surface fallbacks, "
        "and hot-reloading that resolves risk scores into operational routing directives.",
        bold_prefix="3. Dynamic Policy Decision Engine: "
    )
    add_bullet(
        "Iterative counterfactual beam search (beam width k=5, deletion cap L=6) that computes minimal token removal "
        "sets, score reduction delta (Delta s), and safe sanitized text upon BLOCK or REVIEW.",
        bold_prefix="4. Token Attribution & Sanitization Engine: "
    )
    add_bullet(
        "A thread-safe SQLite WAL audit database (cipherguard_audit.db), a production FastAPI REST server (:8000), "
        "and an interactive Streamlit dashboard (:8501).",
        bold_prefix="5. Persistence & Interface Layer: "
    )

    # 3. What Was Built
    add_section_header("3. Inventory of What Was Built (Inception to Phase 1)", 1)
    add_body("The implementation is fully realized in the codebase and verified across multiple operational layers:")
    
    add_section_header("A. Core Backend Modules", 2)
    add_bullet("src/ingestion/: DirectInput and IndirectInput data models with metadata tracking.", bold_prefix="Ingestion: ")
    add_bullet("src/classification/: ModelFamilyA, ModelFamilyB, FinetunedDebertaClassifier, ToxicBertClassifier, PresidioClassifier, and RiskAggregator for unified multi-dimensional score fusion.", bold_prefix="Classifiers: ")
    add_bullet("src/policy/: PolicySchema, PolicyLoader with dynamic hot-reloading, and DecisionEngine.", bold_prefix="Policy: ")
    add_bullet("src/attribution/: Beam search optimization (beam_search.py) and contrastive formatter (explanation.py).", bold_prefix="Attribution: ")
    add_bullet("src/router.py: Master CipherGuardRouter class orchestrating end-to-end routing.", bold_prefix="Orchestrator: ")
    add_bullet("src/logging_store.py: SQLite audit logger recording input text, vectors, decisions, explanations, and latency.", bold_prefix="Audit: ")

    add_section_header("B. Interfaces & Deployments", 2)
    add_bullet("FastAPI REST service (api/main.py) exposing /route, /policy, /logs, and /explain with automated OpenAPI Swagger documentation.", bold_prefix="REST API: ")
    add_bullet("Streamlit dashboard (ui/app.py) featuring interactive prompt testing, real-time attribution token visualizer, live policy editor, and historical audit search.", bold_prefix="Web Dashboard: ")
    add_bullet("Active live deployments on Render (FastAPI) and Streamlit Community Cloud.", bold_prefix="Cloud Deployments: ")

    add_section_header("C. Empirical Benchmarks (Initial Proof-of-Concept, n=85)", 2)
    add_body(
        "The initial proof-of-concept benchmark (Table I and Table II reproductions) established core baseline capabilities:"
    )
    
    t_poc = doc.add_table(rows=5, cols=4)
    t_poc.alignment = WD_TABLE_ALIGNMENT.CENTER
    poc_widths = [Inches(2.2), Inches(1.6), Inches(1.6), Inches(1.6)]
    poc_headers = ["Model / Configuration", "Accuracy", "Recall", "Attribution Metric"]
    poc_rows = [
        ["Model Family A (LogReg, 5x5 CV)", "0.962 (+/- 0.037)", "0.965 (+/- 0.056)", "Diffused Signal (~9.2 tokens)"],
        ["Model Family B (MLP, 5x5 CV)", "0.899 (+/- 0.074)", "0.970 (+/- 0.064)", "High Recall Operating Point"],
        ["DeBERTa-v3 (Zero-Shot)", "0.977", "0.950 (Dir 92%, Ind 100%)", "84.2% Flip Rate @ 2.16 tokens"],
        ["Attribution Beam Search", "--", "--", "Mean CPU Latency: 20-38ms (TFIDF)"]
    ]
    for j, h in enumerate(poc_headers):
        c = t_poc.rows[0].cells[j]
        c.width = poc_widths[j]
        set_cell_background(c, HEX_PRIMARY)
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(poc_rows):
        for j, val in enumerate(row):
            c = t_poc.rows[i+1].cells[j]
            c.width = poc_widths[j]
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 80, 80, 100, 100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4. Phase 1 Accomplishments
    add_section_header("4. Phase 1 Accomplishments: Multi-Label Dataset Pipeline", 1)
    add_body(
        "Phase 1 focused on engineering a robust, multi-label dataset pipeline to replace the initial 85-sample pilot "
        "and establish verified training and evaluation partitions:"
    )

    add_section_header("A. Expanded Dataset Composition (N = 923 Samples)", 2)
    add_body(
        "The expanded dataset aggregates six distinct sources into a uniform multi-label format with five non-mutually exclusive categories:"
    )
    
    t_ds = doc.add_table(rows=8, cols=3)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    ds_widths = [Inches(3.2), Inches(1.2), Inches(2.6)]
    ds_headers = ["Source Repository / Synthetic Generator", "Count", "Role in Dataset"]
    ds_rows = [
        ["deepset/prompt-injections (Hugging Face)", "350", "Public prompt injection benchmark"],
        ["tweet_eval/hate (Hugging Face)", "200", "Hate and toxicity balance (100 hate, 100 benign)"],
        ["cipherguard_maltools_synthetic", "125", "Multi-turn malicious tool attacks & benign controls"],
        ["pilot_dataset_85.json (PoC)", "85", "Original pilot converted to multi-label format"],
        ["cipherguard_pii_synthetic", "99", "4-tier synthetic PII leakage dataset"],
        ["cipherguard_indirect_synthetic", "64", "Indirect attacks (RAG, email, DB, web, tools)"],
        ["Total Combined Dataset", "923", "Full Expanded Multi-Label Corpus"]
    ]
    for j, h in enumerate(ds_headers):
        c = t_ds.rows[0].cells[j]
        c.width = ds_widths[j]
        set_cell_background(c, HEX_SECONDARY)
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(ds_rows):
        for j, val in enumerate(row):
            c = t_ds.rows[i+1].cells[j]
            c.width = ds_widths[j]
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            if i == len(ds_rows) - 1:
                bg = "E2E8F0"
            set_cell_background(c, bg)
            set_cell_margins(c, 80, 80, 100, 100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            r.font.bold = (i == len(ds_rows) - 1)
            r.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_section_header("B. Elimination of Pre-Phase 1 Flaws & Distribution Alignment", 2)
    add_bullet(
        "The legacy 829-sample dataset contained zero PII positives. Phase 1 created 68 positive samples across "
        "a 4-tier taxonomy utilizing demonstrably fake conventions (SSN area code 000, Luhn-failing card numbers, "
        "RFC 2606 .invalid domains, and 555-xxxx phone numbers).",
        bold_prefix="Resolved PII Zero-Positive Defect: "
    )
    add_bullet(
        "Indirect injection samples were expanded from 3.6% (30 samples) to 10.2% (94 samples), "
        "covering RAG context, tool responses, emails, and database records.",
        bold_prefix="Surface Representation Realignment: "
    )
    add_bullet(
        "All 923 records possess full attack_type taxonomic labels and severity tiers (low: 535, medium: 122, high: 130, critical: 136).",
        bold_prefix="Metadata Enrichment (100%): "
    )

    add_section_header("C. Deterministic Stratified Split & Leakage Verification", 2)
    add_bullet(
        "data/train.json (645 samples, 69.9%), data/val.json (138 samples, 15.0%), and data/test.json (140 samples, 15.2%). "
        "Prevalence variance across all five categories is strictly <= 0.7 percentage points.",
        bold_prefix="70 / 15 / 15 Partitioning (Seed 42): "
    )
    add_bullet(
        "The held-out test split is strictly quarantined. No hyperparameter or threshold search may use test feedback.",
        bold_prefix="Held-Out Test Integrity: "
    )
    add_bullet(
        "Pairwise character 3-gram Jaccard similarity testing over ~198,000 comparisons confirmed zero malicious data leakage across splits.",
        bold_prefix="Zero-Leakage Audit: "
    )
    add_bullet(
        "The proof-of-concept dataset (n=85) is locked with SHA-256 (89d6247b9a92ae89...); build scripts abort if modified.",
        bold_prefix="Pilot Checksum Lock: "
    )

    add_section_header("D. Verification Test Suite", 2)
    add_body(
        "The complete automated test suite was executed against the repository. All 13 tests passed with zero failures:"
    )
    add_bullet("test_attribution.py: 2 tests passed (beam search flip verification, search budget termination).")
    add_bullet("test_classification.py: 4 tests passed (Family A, Family B, conservative score fusion, batch scoring).")
    add_bullet("test_policy_engine.py: 3 tests passed (rule resolution, surface-specific threshold matching, fail-safe fallback).")
    add_bullet("test_router.py: 4 tests passed (benign turn pass-through, jailbreak block & attribution, indirect injection, audit logging).")

    # 5. Limitations & Pendings
    add_section_header("5. Identified Limitations, Bottlenecks & Pendings", 1)
    add_body(
        "In accordance with transparent engineering practices, the following constraints and pending tasks are documented:"
    )
    
    t_lim = doc.add_table(rows=6, cols=3)
    t_lim.alignment = WD_TABLE_ALIGNMENT.CENTER
    lim_widths = [Inches(2.0), Inches(2.5), Inches(2.5)]
    lim_headers = ["Identified Limitation", "Root Cause & Impact", "Planned Mitigation"]
    lim_rows = [
        [
            "CPU Attribution Latency (4-6s)",
            "Evaluating beam search deletions iteratively via DeBERTa on CPU incurs latency bottlenecks.",
            "Deploy GPU acceleration, ONNX Runtime INT8 quantization, or two-stage token pre-filtering."
        ],
        [
            "Model Retraining on Train Split",
            "Existing pickle models were initially trained across all 923 samples before split isolation.",
            "Run evaluation/phase2_train.py to retrain Family A and B strictly on data/train.json."
        ],
        [
            "GPU DeBERTa Fine-Tuning",
            "Full 5-category DeBERTa fine-tuning requires 4GB-12GB VRAM and has not yet been executed.",
            "Run LoRA fine-tuning in Google Colab (finetune_deberta.py) and export adapter weights."
        ],
        [
            "Diffused Attribution in Linear Models",
            "TF-IDF models diffuse signal across words, requiring ~9.2 token removals versus 2.16 in DeBERTa.",
            "Use transformer backends when concise, human-interpretable explanations are required."
        ],
        [
            "PowerShell Environment Execution",
            "Running 'pytest' directly without activating the virtual environment causes command not found.",
            "Standardize on .\\.venv\\Scripts\\pytest.exe or activate .\\.venv\\Scripts\\Activate.ps1."
        ]
    ]
    for j, h in enumerate(lim_headers):
        c = t_lim.rows[0].cells[j]
        c.width = lim_widths[j]
        set_cell_background(c, HEX_PRIMARY)
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(lim_rows):
        for j, val in enumerate(row):
            c = t_lim.rows[i+1].cells[j]
            c.width = lim_widths[j]
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 80, 80, 100, 100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 6. What Needs to Be Built Next
    add_section_header("6. Next Phases Roadmap: What Needs to Be Built", 1)
    
    add_section_header("Phase 2: Model Training, Threshold Search & Benchmarking (Immediate Focus)", 2)
    add_bullet("Execute python -m evaluation.phase2_train to fit Model Family A & B exclusively on data/train.json (645 samples).", bold_prefix="1. Clean Train-Split Retraining: ")
    add_bullet("Execute python -m evaluation.finetune_deberta --mode lora on Google Colab or GPU server and save adapter to models/.", bold_prefix="2. GPU LoRA Fine-Tuning: ")
    add_bullet("Execute python -m evaluation.phase2_benchmark to generate PR-AUC, ROC-AUC, F1, Precision, and Recall on data/val.json.", bold_prefix="3. Multi-Model Validation Benchmark: ")
    add_bullet("Execute python -m evaluation.phase2_threshold_search to optimize per-surface thresholds against validation data.", bold_prefix="4. Per-Surface Threshold Optimization: ")
    add_bullet("Execute python -m evaluation.phase2_final_test_eval once against held-out data/test.json to produce official benchmark tables.", bold_prefix="5. Held-Out Test Evaluation: ")

    add_section_header("Phase 3: Production Hardening & Ecosystem Integration", 2)
    add_bullet("Convert transformer classifiers to ONNX Runtime INT8 for sub-50ms CPU inference.", bold_prefix="1. Model Acceleration & Quantization: ")
    add_bullet("Develop plug-and-play middleware interceptors for LangChain, LlamaIndex, and AutoGen pipelines.", bold_prefix="2. Framework Middleware: ")
    add_bullet("Provide streaming token-level inspection for real-time chat completions via WebSockets.", bold_prefix="3. Streaming Ingestion Gateway: ")
    add_bullet("Implement API key authentication, rate limiting, and administrative RBAC for policy modification.", bold_prefix="4. Enterprise Security: ")

    # 7. Quickstart & Instructions
    add_section_header("7. Local Quickstart & Instructions for Teammates", 1)
    add_body("To run and test the complete pipeline locally, execute the following commands:")
    
    commands = [
        ("Activate Virtual Environment", ".\\.venv\\Scripts\\Activate.ps1"),
        ("Execute Full Test Suite", ".\\.venv\\Scripts\\pytest.exe tests/ -v"),
        ("Launch Interactive Dashboard", ".\\.venv\\Scripts\\streamlit.exe run ui/app.py"),
        ("Start REST API Server", ".\\.venv\\Scripts\\uvicorn.exe api.main:app --reload --port 8000"),
        ("Execute Phase 2 Training", ".\\.venv\\Scripts\\python.exe -m evaluation.phase2_train")
    ]
    t_cmd = doc.add_table(rows=len(commands)+1, cols=2)
    t_cmd.alignment = WD_TABLE_ALIGNMENT.CENTER
    cmd_widths = [Inches(2.5), Inches(4.5)]
    for j, h in enumerate(["Target Operation", "Shell Command"]):
        c = t_cmd.rows[0].cells[j]
        c.width = cmd_widths[j]
        set_cell_background(c, HEX_SECONDARY)
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, (op, cmd) in enumerate(commands):
        c0 = t_cmd.rows[i+1].cells[0]
        c1 = t_cmd.rows[i+1].cells[1]
        c0.width = cmd_widths[0]
        c1.width = cmd_widths[1]
        bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
        set_cell_background(c0, bg)
        set_cell_background(c1, bg)
        set_cell_margins(c0, 60, 60, 90, 90)
        set_cell_margins(c1, 60, 60, 90, 90)
        
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(op)
        r0.font.size = Pt(8.5)
        r0.font.color.rgb = COLOR_DARK
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(cmd)
        r1.font.name = "Consolas"
        r1.font.size = Pt(8.5)
        r1.font.color.rgb = COLOR_PRIMARY

    doc.save(filename)
    print(f"DOCX successfully generated: {filename}")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Top rule
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 40, letter[0] - 54, 40)
        # Footer text
        self.drawString(54, 28, "CipherGuard: Policy-Aware Safety Router | Project Status Report")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 28, page_text)
        self.restoreState()


def generate_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F2942")
    c_secondary = colors.HexColor("#1E5F8A")
    c_dark = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")
    c_light_bg = colors.HexColor("#F1F5F9")
    c_border = colors.HexColor("#CBD5E1")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    code_text = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=c_primary
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("CipherGuard: Project Status Report", title_style))
    story.append(Paragraph("System Architecture, Inception to Phase 1 Milestone, Findings, and Next Phases Roadmap", subtitle_style))

    # Metadata Strip Box
    meta_table_data = [
        [
            Paragraph("<b>CURRENT MILESTONE</b><br/>Phase 1 Complete (Dataset & Baseline)", table_text),
            Paragraph("<b>SYSTEM STATUS</b><br/>Live REST API & UI Active", table_text),
            Paragraph("<b>TEST VERIFICATION</b><br/>13 / 13 Tests Passing (100%)", table_text)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[168, 168, 168])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light_bg),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Motivation", h1_style))
    story.append(Paragraph(
        "Modern Large Language Model (LLM) agent workflows face severe security vulnerabilities from direct user "
        "jailbreaks (DPI) and indirect prompt injections (IPI) embedded in external retrieved context (RAG), database "
        "records, tool outputs, and incoming emails. Existing safety solutions suffer from two fundamental bottlenecks:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Monolithic Classifier Rigidity (e.g., Llama Guard, ShieldGemma):</b> Hardcodes moderation policies directly "
        "into neural weights; altering safety risk tolerance requires computationally expensive model retraining or fine-tuning.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>LLM-as-a-Judge Overhead:</b> Invoking secondary generative LLMs to evaluate input safety incurs high per-token "
        "API costs, adds hundreds of milliseconds of latency, and yields non-deterministic, unauditable decisions.",
        bullet_style
    ))
    story.append(Paragraph(
        "CipherGuard bridges this gap by providing an inline, auditable pre-agent safety gateway characterized by "
        "three primary technical innovations:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Dynamic Policy Decoupling:</b> A hot-reloadable, declarative JSON policy matrix that decouples runtime routing "
        "decisions (BLOCK, REVIEW, ALLOW) from classifier weights, enabling zero-downtime threshold adjustments.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Dual-Surface Threat Modeling:</b> Independent risk scoring and threshold calibration tailored to the distinct "
        "operational characteristics of user turns ('direct') versus ingested third-party data ('indirect').",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Removal-Based Contrastive Attribution:</b> An approximate beam search engine that isolates the minimal subset of "
        "trigger tokens whose removal flips a rejection to a pass, generating both verifiable audit rationale and safe sanitized text.",
        bullet_style
    ))

    # 2. System Architecture
    story.append(Paragraph("2. System Architecture & Subsystems", h1_style))
    story.append(Paragraph(
        "CipherGuard intercepts untrusted payloads before they reach core LLMs. The architecture is partitioned into five decoupled subsystems:",
        body_style
    ))
    story.append(Paragraph("• <b>1. Ingestion & Normalization:</b> DirectInput and IndirectInput wrappers tracking surface provenance across RAG, tools, emails, and web data.", bullet_style))
    story.append(Paragraph("• <b>2. Classification & Risk Fusion:</b> Model Family A (LogReg), Family B (MLP), DeBERTa-v3 semantic classifier, ToxicBERT, and Presidio aggregated via conservative max fusion.", bullet_style))
    story.append(Paragraph("• <b>3. Dynamic Policy Decision Engine:</b> Pydantic-validated rules engine evaluating JSON policies with wildcard fallbacks and zero-restart hot reloading.", bullet_style))
    story.append(Paragraph("• <b>4. Token Attribution Engine:</b> Iterative beam search (k=5, L=6) computing minimal trigger tokens, score reduction delta (Delta s), and safe sanitized text.", bullet_style))
    story.append(Paragraph("• <b>5. Persistence & Interfaces:</b> SQLite WAL audit store (cipherguard_audit.db), FastAPI REST gateway (:8000), and interactive Streamlit UI (:8501).", bullet_style))

    # 3. What Was Built
    story.append(Paragraph("3. Inventory of What Was Built (Inception to Phase 1)", h1_style))
    story.append(Paragraph("<b>A. Core Backend Modules</b>", h2_style))
    story.append(Paragraph("• <b>Ingestion:</b> src/ingestion/ implements DirectInput and IndirectInput with metadata provenance.", bullet_style))
    story.append(Paragraph("• <b>Classification:</b> src/classification/ contains ModelFamilyA, ModelFamilyB, FinetunedDebertaClassifier, ToxicBertClassifier, PresidioClassifier, and RiskAggregator.", bullet_style))
    story.append(Paragraph("• <b>Policy Engine:</b> src/policy/ provides PolicySchema, dynamic PolicyLoader, and multi-surface DecisionEngine.", bullet_style))
    story.append(Paragraph("• <b>Attribution:</b> src/attribution/ executes batched beam search (beam_search.py) and builds counterfactual summaries (explanation.py).", bullet_style))
    story.append(Paragraph("• <b>Gateway & Logging:</b> src/router.py orchestrates pipeline flow, and src/logging_store.py writes structured SQLite audit records.", bullet_style))

    story.append(Paragraph("<b>B. Interfaces & Deployments</b>", h2_style))
    story.append(Paragraph("• <b>REST API:</b> FastAPI server (api/main.py) with /route, /policy, /logs, and /explain endpoints.", bullet_style))
    story.append(Paragraph("• <b>Interactive UI:</b> Streamlit dashboard (ui/app.py) for testing prompts, visual token attribution, live policy tuning, and audit log analysis.", bullet_style))
    story.append(Paragraph("• <b>Cloud Deployments:</b> Deployed on Render (FastAPI REST service) and Streamlit Community Cloud.", bullet_style))

    story.append(Paragraph("<b>C. Initial Proof-of-Concept Benchmarks (n=85 Pilot Dataset)</b>", h2_style))
    
    poc_data = [
        [Paragraph("Model / Configuration", table_header), Paragraph("Accuracy", table_header), Paragraph("Recall", table_header), Paragraph("Attribution Metric", table_header)],
        [Paragraph("Model Family A (LogReg, 5x5 CV)", table_text), Paragraph("0.962 (+/- 0.037)", table_text), Paragraph("0.965 (+/- 0.056)", table_text), Paragraph("Diffused Signal (~9.2 tokens)", table_text)],
        [Paragraph("Model Family B (MLP, 5x5 CV)", table_text), Paragraph("0.899 (+/- 0.074)", table_text), Paragraph("0.970 (+/- 0.064)", table_text), Paragraph("High Recall Operating Point", table_text)],
        [Paragraph("DeBERTa-v3 (Zero-Shot)", table_text), Paragraph("0.977", table_text), Paragraph("0.950 (Dir 92%, Ind 100%)", table_text), Paragraph("84.2% Flip Rate @ 2.16 tokens", table_text)],
        [Paragraph("Attribution Beam Search", table_text), Paragraph("--", table_text), Paragraph("--", table_text), Paragraph("Mean CPU Latency: 20-38ms (TFIDF)", table_text)]
    ]
    t_poc_flow = Table(poc_data, colWidths=[150, 110, 110, 134])
    t_poc_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_poc_flow)
    story.append(Spacer(1, 8))

    # 4. Phase 1 Accomplishments
    story.append(Paragraph("4. Phase 1 Accomplishments: Multi-Label Dataset Pipeline", h1_style))
    story.append(Paragraph("<b>A. Expanded Dataset Composition (N = 923 Samples)</b>", h2_style))
    
    ds_data = [
        [Paragraph("Source Repository / Synthetic Generator", table_header), Paragraph("Count", table_header), Paragraph("Role in Dataset", table_header)],
        [Paragraph("deepset/prompt-injections (Hugging Face)", table_text), Paragraph("350", table_text), Paragraph("Public prompt injection benchmark", table_text)],
        [Paragraph("tweet_eval/hate (Hugging Face)", table_text), Paragraph("200", table_text), Paragraph("Hate and toxicity balance (100 hate, 100 benign)", table_text)],
        [Paragraph("cipherguard_maltools_synthetic", table_text), Paragraph("125", table_text), Paragraph("Multi-turn malicious tool attacks & benign controls", table_text)],
        [Paragraph("pilot_dataset_85.json (PoC)", table_text), Paragraph("85", table_text), Paragraph("Original pilot converted to multi-label format", table_text)],
        [Paragraph("cipherguard_pii_synthetic", table_text), Paragraph("99", table_text), Paragraph("4-tier synthetic PII leakage dataset", table_text)],
        [Paragraph("cipherguard_indirect_synthetic", table_text), Paragraph("64", table_text), Paragraph("Indirect attacks (RAG, email, DB, web, tools)", table_text)],
        [Paragraph("<b>Total Combined Dataset</b>", table_text), Paragraph("<b>923</b>", table_text), Paragraph("<b>Full Expanded Multi-Label Corpus</b>", table_text)]
    ]
    t_ds_flow = Table(ds_data, colWidths=[204, 80, 220])
    t_ds_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_ds_flow)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>B. Elimination of Pre-Phase 1 Flaws & Split Verification</b>", h2_style))
    story.append(Paragraph(
        "• <b>Resolved PII Zero-Positive Defect:</b> The pre-Phase 1 dataset contained 0 PII positives. Phase 1 created 68 positive "
        "samples across a 4-tier taxonomy using demonstrably fake conventions (SSN area code 000, Luhn-failing cards, RFC 2606 .invalid domains).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Enhanced Indirect Surface:</b> Indirect attack representation increased from 3.6% (30 samples) to 10.2% (94 samples).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Metadata Taxonomy:</b> 100% of samples possess attack_type and severity classifications (low: 535, med: 122, high: 130, crit: 136).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Stratified 70/15/15 Split:</b> Train: 645 (69.9%), Val: 138 (15.0%), Held-out Test: 140 (15.2%) with prevalence variance <= 0.7%.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Zero-Leakage Confirmation:</b> Character 3-gram Jaccard similarity audit (~198,000 comparisons) confirmed zero data leakage across splits.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>100% Test Suite Verification:</b> All 13 unit and integration tests passed with zero failures in 57.38 seconds.",
        bullet_style
    ))

    # 5. Limitations & Pendings
    story.append(Paragraph("5. Identified Limitations, Bottlenecks & Pendings", h1_style))
    lim_data = [
        [Paragraph("Identified Limitation", table_header), Paragraph("Root Cause & Impact", table_header), Paragraph("Planned Mitigation", table_header)],
        [
            Paragraph("CPU Attribution Latency (4-6s)", table_text),
            Paragraph("Beam search deletions iteratively pass through DeBERTa on CPU, adding latency.", table_text),
            Paragraph("Deploy GPU acceleration, ONNX Runtime INT8 quantization, or token pre-filtering.", table_text)
        ],
        [
            Paragraph("Model Retraining on Train Split", table_text),
            Paragraph("Existing pickle models were initially trained across all 923 samples before split isolation.", table_text),
            Paragraph("Run evaluation/phase2_train.py to retrain Family A & B strictly on data/train.json.", table_text)
        ],
        [
            Paragraph("GPU DeBERTa Fine-Tuning", table_text),
            Paragraph("Full 5-category DeBERTa fine-tuning requires 4GB-12GB VRAM and is pending GPU run.", table_text),
            Paragraph("Run LoRA fine-tuning in Google Colab (finetune_deberta.py) and export weights.", table_text)
        ],
        [
            Paragraph("Diffused Attribution in Linear Models", table_text),
            Paragraph("TF-IDF models diffuse signal, requiring ~9.2 token removals vs 2.16 in DeBERTa.", table_text),
            Paragraph("Use transformer backends when concise, human-interpretable explanations are required.", table_text)
        ],
        [
            Paragraph("PowerShell Virtualenv Pathing", table_text),
            Paragraph("Executing 'pytest' without activated virtual environment triggers command not found.", table_text),
            Paragraph("Standardize on .\\.venv\\Scripts\\pytest.exe or activate virtualenv first.", table_text)
        ]
    ]
    t_lim_flow = Table(lim_data, colWidths=[140, 180, 184])
    t_lim_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_lim_flow)
    story.append(Spacer(1, 8))

    # 6. Next Phases Roadmap
    story.append(Paragraph("6. Next Phases Roadmap: What Needs to Be Built", h1_style))
    story.append(Paragraph("<b>Phase 2: Training, Optimization & Validation Benchmarking (Immediate)</b>", h2_style))
    story.append(Paragraph("• <b>Clean Train-Split Retraining:</b> Execute python -m evaluation.phase2_train on data/train.json.", bullet_style))
    story.append(Paragraph("• <b>GPU LoRA Fine-Tuning:</b> Run python -m evaluation.finetune_deberta --mode lora on Colab / GPU server.", bullet_style))
    story.append(Paragraph("• <b>Validation Benchmarking:</b> Run python -m evaluation.phase2_benchmark on data/val.json across all models.", bullet_style))
    story.append(Paragraph("• <b>Threshold Optimization:</b> Execute python -m evaluation.phase2_threshold_search to tune policy.json operating points.", bullet_style))
    story.append(Paragraph("• <b>Final Test Evaluation:</b> Run python -m evaluation.phase2_final_test_eval once on held-out data/test.json.", bullet_style))

    story.append(Paragraph("<b>Phase 3: Production Hardening & Ecosystem Integration</b>", h2_style))
    story.append(Paragraph("• <b>Model Quantization:</b> Convert DeBERTa and ToxicBERT to ONNX Runtime INT8 for sub-50ms CPU latency.", bullet_style))
    story.append(Paragraph("• <b>Framework Adapters:</b> Build middleware plugins for LangChain and LlamaIndex agent pipelines.", bullet_style))
    story.append(Paragraph("• <b>Streaming Ingestion:</b> Implement WebSocket token-level inspection for real-time chat completions.", bullet_style))
    story.append(Paragraph("• <b>Enterprise RBAC:</b> Secure policy endpoints with API key management and role-based administration.", bullet_style))

    # 7. Quickstart
    story.append(Paragraph("7. Local Quickstart for Teammates", h1_style))
    cmd_data = [
        [Paragraph("Target Operation", table_header), Paragraph("Command Line", table_header)],
        [Paragraph("Activate Virtual Environment", table_text), Paragraph(".\\.venv\\Scripts\\Activate.ps1", code_text)],
        [Paragraph("Execute Full Test Suite", table_text), Paragraph(".\\.venv\\Scripts\\pytest.exe tests/ -v", code_text)],
        [Paragraph("Launch Interactive Dashboard", table_text), Paragraph(".\\.venv\\Scripts\\streamlit.exe run ui/app.py", code_text)],
        [Paragraph("Start REST API Server", table_text), Paragraph(".\\.venv\\Scripts\\uvicorn.exe api.main:app --reload --port 8000", code_text)],
        [Paragraph("Execute Phase 2 Training", table_text), Paragraph(".\\.venv\\Scripts\\python.exe -m evaluation.phase2_train", code_text)]
    ]
    t_cmd_flow = Table(cmd_data, colWidths=[180, 324])
    t_cmd_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd_flow)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {filename}")


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    
    docx_path = "docs/CipherGuard_Phase1_Project_Status_Report.docx"
    pdf_path = "docs/CipherGuard_Phase1_Project_Status_Report.pdf"
    
    generate_docx(docx_path)
    generate_pdf(pdf_path)
    
    generate_docx("CipherGuard_Phase1_Project_Status_Report.docx")
    generate_pdf("CipherGuard_Phase1_Project_Status_Report.pdf")
