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
| **Safe Auto-Handle Precision** | `0.4800` | `0.5429` | **`0.9495`** | **Hero Metric**: when saying Auto-Handle, is it truly safe? |
| **Escalation Recall** | `0.0000` | `0.3846` | **`0.9519`** | **Hero Metric**: coverage of critical security, fraud, and theft inquiries |
| **Missed Escalation Rate** | `1.0000` | `0.6154` | **`0.0481`** | **Critical Safety Failure**: true risk queries dangerously automated |
| **False Escalation Rate** | `0.0000` | `0.2083` | **`0.0208`** | Over-escalation rate (human agent queue bloat & cost) |
| **Intent Macro-F1** | `0.0227` | `0.9039` | **`0.9725`** | Unskewed multi-class classification metric |
| **Intent Overall Accuracy** | `0.1000` | `0.9000` | **`0.9750`** | Classification correctness across all 8 intents |
| **Escalation Precision** | `0.0000` | `0.6667` | **`0.9802`** | Proportion of escalated queries that legitimately require humans |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.8650`** | Deterministic 4-Criteria Rubric Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | 100% compliance with deterministic PII-safety rules (zero credential solicitation) |

### Key Finding
> **Deterministic safety guardrails prevent operational disasters.**  
> While simple keyword models miss over **61% of escalations**, our deterministic triage engine reduces the Missed Escalation Rate to **4.81%**, substantially improving coverage of stolen-package, account-security, and financial-risk cases.

### Biggest Limitation
> **Historical data is evidence of past behavior, not current policy.**  
> The Twitter Customer Support dataset reflects 2017 operating conditions. Modern Amazon workflows rely on authenticated in-app handoffs that cannot be verified solely from public historical tweets. Furthermore, the adversarial tier has materially lower Macro-F1 than the normal tier (0.7891 vs. 0.9796); qualitative failure analysis suggests sarcasm, hostility, retrospective praise, and ambiguous complaint phrasing as contributing factors. [Read the full Sampling & Hand-Labeling Note](data/golden/LABELING_NOTE.md).

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

### 2. Run the Automated Pytest Suite (21 Verification Tests)
```bash
python -m pytest tests/ -v
```
Executes in ~5 seconds with zero external network calls: verifies zero data leakage, sub-100ms retrieval latency SLA, taxonomy collision precedence, reply sanitization, and rubric scoring.

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
python -m src.cli "My package says delivered yesterday but it was never left on my porch!"
```

**Output**:
```json
{
  "intent": "DAMAGED_WRONG_MISSING",
  "intent_confidence": 0.88,
  "decision": "ESCALATE",
  "escalation_category": "LOST_OR_STOLEN_DELIVERY",
  "reason": "Deterministic safety rule triggered: lost or stolen delivery detected in customer message.",
  "reply": "I'm so sorry to hear your package hasn't turned up even though it's marked as delivered! For your privacy, please do not post your order details here. Please send us a direct message with your order number and email address through our secure link [link] so we can investigate with the carrier right away. ^SP",
  "evidence": [
    {
      "conversation_id": "2914a1fb2db85fc284151b3c52462070",
      "similarity": 0.5435
    }
  ]
}
```

---

## 🏗️ System Architecture

```
                  CUSTOMER TWEET
                       │
                       ▼
                 PREPROCESSING
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         Intent     Risk       Hybrid
       Classifier  Rules     Retrieval
            │          │      (55k KB)
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
        (Retrieval-Conditioned Brand Voice)
                        │
                        ▼
               RESPONSE SANITIZER
              (PII & Channel Guard)
                        │
                        ▼
                   FINAL JSON
```

---

## 📂 Repository Structure

```
HIVER/
├── README.md                      # Marketing page, quickstart, and headline benchmark table
├── REPORT.md                      # Comprehensive 6-page technical report with mandatory sections
├── DECISION_LOG.md                # 16 non-obvious engineering decisions and their rationales
├── requirements.txt               # Lightweight Python dependencies (including pytest)
├── run_pipeline.py                # Master reproduction script (typically < 1 minute)
│
├── tests/                         # Full automated test suite (21 unit & safety tests)
│   ├── test_agent.py              # End-to-end agent behavior, routing & credential sanitization
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
│   ├── agent.py                   # Production-minded AI Support Agent with response sanitizer
│   ├── baselines.py               # Baseline 0 (Trivial) and Baseline 1 (Simple)
│   ├── evaluator.py               # Macro-F1, safety triage metrics, and confusion plotting
│   ├── llm_judge.py               # Deterministic 4-criteria reply quality rubric
│   ├── failure_analysis.py        # Automated isolation of real edge failures
│   ├── audit_and_fix.py           # Independent ground-truth relabeling and taxonomy audit
│   └── cli.py                     # Interactive terminal console for live testing
│
└── artifacts/
    ├── evaluation_metrics.json    # Complete JSON dump of all computed metrics
    ├── headline_results_table.md  # Clean Markdown benchmark summary
    ├── failure_analysis.json      # Structured log of all failure edge cases
    ├── confusion_matrix_proposed_agent.png
    ├── confusion_matrix_simple_baseline.png
    └── confusion_matrix_trivial_baseline.png
```

---

## 📑 Core Documentation Links
- **[Full 6-Page Technical Report](REPORT.md)**: In-depth problem framing, multi-brand audit scoring matrix, failure analysis, and the mandatory *"What is misleading about my headline number?"* critique.
- **[Engineering Decision Log](DECISION_LOG.md)**: 16 non-obvious architectural decisions, trade-offs, and design rationales.
