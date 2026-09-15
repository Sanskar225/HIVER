# Hiver SDE Intern Take-Home: AI Customer Support Agent & Evaluation Harness

> **Brand Selected**: `@AmazonHelp` (E-Commerce & Digital Logistics)  
> **System Status**: Evaluation-First Support-Agent Prototype (Production-Minded)  
> **Core Operating Philosophy**: *"The objective is not maximum automation. The objective is maximum safe resolution."*  
> **Reproducibility**: Entire evaluation harness executes in **under 1 minute** locally with zero external API dependencies.

---

## TL;DR (30-Second Overview)

We built an evaluation-first AI customer support prototype for **`@AmazonHelp`** evaluated on an independently hand-verified **Golden Suite of 200 stratified test cases** (140 Normal, 40 Difficult, 20 Adversarial) against **two baselines** (a Trivial Majority-class baseline and a Simple TF-IDF + Keyword model).

### Headline Benchmark Results

| Evaluation Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Safe Auto-Handle Precision** | `0.4800` | `0.5429` | **`0.8947`** | **Hero Metric**: when saying Auto-Handle, is it truly safe? |
| **Escalation Recall** | `0.0000` | `0.3846` | **`0.9038`** | **Hero Metric**: coverage of critical security, fraud, and theft inquiries |
| **Missed Escalation Rate** | `1.0000` | `0.6154` | **`0.0962`** | **Critical Safety Failure**: true risk queries dangerously automated |
| **False Escalation Rate** | `0.0000` | `0.2083` | **`0.1146`** | Over-escalation rate (human agent queue bloat & cost) |
| **Intent Macro-F1** | `0.0227` | `0.9039` | **`0.8441`** | Unskewed multi-class classification metric |
| **Intent Overall Accuracy** | `0.1000` | `0.9000` | **`0.8700`** | Classification correctness across all 8 intents |
| **Escalation Precision** | `0.0000` | `0.6667` | **`0.8952`** | Cleanliness of human triage queue |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.9000`** | Deterministic 4-Criteria Rubric Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | 100% compliance with deterministic PII-safety rules (zero credential solicitation) |

### Key Finding
> **Calibrated statistical ML combined with deterministic safety guardrails prevents operational disasters.**  
> While simple keyword models miss over **61% of escalations**, our deterministic triage engine backed by a mathematically calibrated statistical ML model (`CalibratedClassifierCV` with Platt scaling) reduces the Missed Escalation Rate to **9.62%** with an exceptional **89.52% Escalation Precision** and an Expected Calibration Error (ECE) of **0.0809**, ensuring human supervisor queues are protected from out-of-distribution hallucinations and false alarms.

### Biggest Limitation
> **Historical data is evidence of past behavior, not current policy.**  
> The Twitter Customer Support dataset reflects 2017 operating conditions. Modern Amazon workflows rely on authenticated in-app handoffs that cannot be verified solely from public historical tweets. Furthermore, adversarial tier performance requires continuous monitoring against sarcasm and linguistic ambiguity. [Read the full Sampling & Hand-Labeling Note](data/golden/LABELING_NOTE.md).

---

## 🚀 Quickstart (Reproduce Headline Results in < 15 Minutes)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Sanskar225/HIVER.git
cd HIVER

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Automated Pytest Suite (35 Verification Tests)
```bash
python -m pytest tests/ -v
```
Executes in ~20 seconds with zero external network calls: verifies zero data leakage, sub-100ms retrieval latency SLA, taxonomy collision precedence, negation handling, calibrated confidence gating on OOD inputs, reply sanitization, multi-turn continuity, and FastAPI endpoints.

### 3. Run the Full Evaluation Pipeline (Typically < 1 Minute)
```bash
python run_pipeline.py
```
This executes predictions across all 200 Golden Evaluation examples for Baseline 0, Baseline 1, and the Proposed Agent, evaluates deterministic reply quality rubric criteria, and outputs:
- `artifacts/evaluation_metrics.json`
- `artifacts/headline_results_table.md`
- `artifacts/confusion_matrix_*.png`

### 4. Interactive CLI Demo (Try It Live!)
```bash
python -m src.cli "Package says delivered yesterday but it was never left on my porch!"
```

**Output**:
```json
{
  "intent": "DAMAGED_WRONG_MISSING",
  "intent_confidence": 0.9187,
  "decision": "ESCALATE",
  "escalation_category": "LOST_OR_STOLEN_DELIVERY",
  "reason": "Deterministic safety rule triggered: lost or stolen delivery detected in customer message.",
  "reply": "I'm so sorry to hear your package hasn't turned up even though it's marked as delivered! For your privacy, please do not post your order details here. Please send us a direct message with your order number and email address through our secure link [link] so we can investigate with the carrier right away. ^MT",
  "evidence": [
    {
      "conversation_id": "8ae129823a0ea3323658dba53a7c30cb",
      "similarity": 0.617
    }
  ]
}
```

### 5. Launch Production FastAPI Microservice
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI/Swagger documentation available at `http://localhost:8000/docs`. Supports stateful multi-turn conversation tracking (`/v1/chat`), high-speed queue routing (`/v1/triage`), and system health status (`/health`).

---

## 🏗️ System Architecture

```
                  CUSTOMER TWEET / CHAT
                           │
                           ▼
                     PREPROCESSING
                           │
                ┌──────────┼──────────┐
                ▼          ▼          ▼
             Calibrated  Deterministic  Hybrid
             Statistical  Safety Rules  Retrieval
             ML Model    (Zero Over-   (55k KB)
           (P(y|x) probs)   ride)
                │          │          │
                └──────────┼──────────┘
                           ▼
                     TRIAGE ENGINE
              ("Maximize Safe Resolution,
                 Not Automation Rate")
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
             AUTO_HANDLE         ESCALATE
                  │                 │
                  └────────┬────────┘
                           ▼
                    REPLY GENERATOR
           (Functional Retrieval Synthesis)
                           │
                           ▼
                   RESPONSE SANITIZER
                  (PII & Channel Guard)
                           │
                           ▼
                    FINAL JSON / API
```

---

## 📂 Repository Structure

```
HIVER/
├── README.md                      # Marketing page, quickstart, and headline benchmark table
├── REPORT.md                      # Comprehensive 6-page technical report with mandatory sections
├── DECISION_LOG.md                # 16 non-obvious engineering decisions and their rationales
├── requirements.txt               # Lightweight Python dependencies (including pytest, fastapi)
├── run_pipeline.py                # Master reproduction script (typically < 1 minute)
│
├── tests/                         # Full automated test suite (31 unit & safety tests)
│   ├── test_agent.py              # End-to-end agent behavior, routing & credential sanitization
│   ├── test_api.py                # FastAPI endpoints (/health, /v1/triage, /v1/chat)
│   ├── test_data_leakage.py       # Zero conversation ID & customer text overlap checks
│   ├── test_evaluator.py          # Metric calculations & deterministic reply rubric tests
│   ├── test_retriever.py          # TF-IDF retrieval accuracy & sub-100ms latency SLA
│   └── test_taxonomy.py           # Hierarchy priority & collision resolution checks
│
├── data/
│   ├── raw/                       # Original multi-turn conversation dataset
│   ├── processed/
│   │   ├── kb_corpus.parquet      # 55,011 historical @AmazonHelp resolution pairs (leak-free)
│   │   └── candidate_eval_pool    # Candidate evaluation sampling pool (unverified)
│   └── golden/
│       ├── golden_eval_set.json   # 200 hand-verified stratified evaluation cases
│       └── golden_eval_set.csv    # Tabular CSV export of golden evaluation suite
│
├── src/
│   ├── config.py                  # Taxonomy definitions, paths, and constants
│   ├── data_prep.py               # Leakage-free dataset parsing and partitioning
│   ├── taxonomy.py                # 8-intent definitions, priority hierarchy & safety rules
│   ├── retriever.py               # Fast TF-IDF historical case search index (<100ms SLA)
│   ├── agent.py                   # Calibrated ML AI Support Agent with stateful memory
│   ├── api.py                     # Production FastAPI service with thread-safe session store
│   ├── baselines.py               # Baseline 0 (Trivial) and Baseline 1 (Simple)
│   ├── evaluator.py               # Macro-F1, safety triage metrics, and confusion plotting
│   ├── quality_rubric.py          # Deterministic 4-criteria reply quality rubric
│   ├── failure_analysis.py        # Automated isolation of real edge failures
│   └── cli.py                     # Interactive terminal console for live testing
│
├── scripts/
│   ├── build_golden_set.py        # Offline golden evaluation benchmark builder
│   └── audit_and_fix.py           # Offline ground-truth relabeling and taxonomy audit
│
└── artifacts/
    ├── evaluation_metrics.json    # Complete JSON dump of all computed metrics
    ├── headline_results_table.md  # Clean Markdown benchmark summary
    ├── agent_intent_model.pkl     # Cached calibrated statistical intent ML classifier
    ├── failure_analysis.json      # Structured log of all failure edge cases
    ├── confusion_matrix_proposed_agent.png
    ├── confusion_matrix_simple_baseline.png
    └── confusion_matrix_trivial_baseline.png
```

---

## 📑 Core Documentation Links
- **[Full 6-Page Technical Report](REPORT.md)**: In-depth problem framing, multi-brand audit scoring matrix, failure analysis, and the mandatory *"What is misleading about my headline number?"* critique.
- **[Engineering Decision Log](DECISION_LOG.md)**: 16 non-obvious architectural decisions, trade-offs, and design rationales.
