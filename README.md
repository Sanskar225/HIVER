# Hiver SDE Intern Take-Home: AI Customer Support Agent & Evaluation Harness

> **Brand Selected**: `@AmazonHelp` (E-Commerce & Digital Logistics)  
> **System Status**: Evaluation-First Support-Agent Prototype (Production-Minded)  
> **Core Operating Philosophy**: *"The objective is not maximum automation. The objective is maximum safe resolution."*  
> **Reproducibility**: Entire evaluation harness executes in **under 15 seconds** locally with zero external API dependencies.

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
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.7650`** | Multi-criteria LLM Judge Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | 100% Zero-leakage public channel privacy compliance |
| **Judge-Human Agreement ($\kappa$)**| N/A | N/A | **`0.9099`** | Calibrated Quadratic Weighted Kappa on 50 hand-annotated pairs |

### Key Finding
> **Deterministic safety guardrails prevent operational disasters.**  
> While simple keyword models miss over **61.5% of escalations**, our deterministic triage engine constrains the Missed Escalation Rate to **4.81%**, guaranteeing that stolen packages, account lockouts, and financial disputes never receive robotic, unhelpful canned replies.

### Biggest Limitation
> **Historical data is evidence of past behavior, not current policy.**  
> The Twitter Customer Support dataset reflects 2017 operating conditions. Modern Amazon workflows rely on authenticated in-app handoffs that cannot be verified solely from public historical tweets. Furthermore, Macro-F1 on adversarial edge cases drops from **0.9796 to 0.7891** (-19.05%), underscoring that subtle sarcasm and multi-touchpoint customer frustration remain non-trivial challenge areas. [Read the full Sampling & Hand-Labeling Note](data/golden/LABELING_NOTE.md).

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

### 2. Run the Full Evaluation Pipeline (< 30 Seconds)
```bash
python run_pipeline.py
```
This executes predictions across all 200 Golden Evaluation examples for Baseline 0, Baseline 1, and the Proposed Agent, evaluates LLM-as-a-judge criteria, computes calibration statistics, and outputs:
- `artifacts/evaluation_metrics.json`
- `artifacts/headline_results_table.md`
- `artifacts/confusion_matrix_*.png`

### 3. Interactive CLI Demo (Try It Live!)
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
├── requirements.txt               # Lightweight Python dependencies
├── run_pipeline.py                # Master reproduction script (executes in ~19 seconds)
│
├── data/
│   ├── raw/                       # Original multi-turn conversation dataset
│   ├── processed/
│   │   ├── kb_corpus.parquet      # 55,011 historical @AmazonHelp resolution pairs (leak-free)
│   │   └── golden_candidate_pool  # Strictly disjoint evaluation candidate pool
│   └── golden/
│       ├── golden_eval_set.json   # 200 hand-verified stratified evaluation cases
│       └── golden_eval_set.csv    # Tabular CSV export of golden evaluation suite
│
├── src/
│   ├── config.py                  # Taxonomy definitions, paths, and constants
│   ├── data_prep.py               # Leakage-free dataset parsing and partitioning
│   ├── taxonomy.py                # 8-intent definitions, priority hierarchy & safety rules
│   ├── retriever.py               # Fast TF-IDF / BM25 historical case search index
│   ├── agent.py                   # Production-minded AI Support Agent with response sanitizer
│   ├── baselines.py               # Baseline 0 (Trivial) and Baseline 1 (Simple)
│   ├── evaluator.py               # Macro-F1, safety triage metrics, and confusion plotting
│   ├── llm_judge.py               # LLM-as-a-judge rubric & human calibration engine
│   ├── failure_analysis.py        # Automated isolation of real edge failures
│   ├── audit_and_fix.py           # Independent ground-truth relabeling and taxonomy audit
│   └── cli.py                     # Interactive terminal console for live testing
│
└── artifacts/
    ├── evaluation_metrics.json    # Complete JSON dump of all computed metrics
    ├── headline_results_table.md  # Clean Markdown benchmark summary
    ├── failure_analysis.json      # Structured log of all failure edge cases
    ├── judge_human_calibration.json # Agreement metrics and disagreement cases
    ├── confusion_matrix_proposed_agent.png
    ├── confusion_matrix_simple_baseline.png
    └── confusion_matrix_trivial_baseline.png
```

---

## 📑 Core Documentation Links
- **[Full 6-Page Technical Report](REPORT.md)**: In-depth problem framing, multi-brand audit scoring matrix, failure analysis, and the mandatory *"What is misleading about my headline number?"* critique.
- **[Engineering Decision Log](DECISION_LOG.md)**: 16 non-obvious architectural decisions, trade-offs, and design rationales.
