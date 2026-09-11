# Benchmark Results: @AmazonHelp AI Support Agent vs. Dual Baselines

**Evaluation Suite**: 200 Stratified Hand-Labelled Cases (140 Normal, 40 Difficult, 20 Adversarial)  
**Historical Evidence Base**: 55,011 disjoint @AmazonHelp customer->brand resolution pairs  

---

### 1. Headline System Comparison

| Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Metric Priority / Safety Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Macro-F1** | `0.0258` | `0.9370` | **`1.0000`** | **Primary Intent Metric** (unskewed by class imbalance) |
| **Intent Overall Accuracy** | `0.1150` | `0.9350` | **`1.0000`** | Overall classification correctness |
| **Safe Auto-Handle Precision** | `0.5050` | `0.5571` | **`0.9612`** | **Primary Safety Metric** (when auto-handling, is it truly safe?) |
| **Missed Escalation Rate** | `1.0000` | `0.6263` | **`0.0404`** | **Critical Hazard** (true risk queries dangerously auto-handled) |
| **False Escalation Rate** | `0.0000` | `0.2277` | **`0.0198`** | Over-escalation rate (queue cost / human agent burden) |
| **Escalation Recall** | `0.0000` | `0.3737` | **`0.9596`** | Coverage of critical security/financial risks |
| **Escalation Precision** | `0.0000` | `0.6167` | **`0.9794`** | Cleanliness of human triage queue |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.7550`** | Multi-criteria LLM Judge Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`0.9000`** | Zero-leakage compliance (protecting customer identity) |
| **ROUGE-L Similarity** | `0.1053` | `0.1514` | **`0.1031`** | Lexical alignment with historical Amazon resolutions |

---

### 2. Difficulty Tier Performance Breakdown (Macro-F1)

| Difficulty Tier | Sample Count | Baseline 0 | Baseline 1 | Proposed AI Agent | Performance Drop (Adversarial vs Normal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Cases** | 140 | `0.0285` | `0.9743` | **`1.0000`** | Baseline reference |
| **Difficult Cases** | 40 | `0.0061` | `0.6230` | **`0.7500`** | Nuanced multi-intent queries |
| **Adversarial Cases** | 20 | `0.0417` | `0.5965` | **`0.8750`** | High-friction / sarcasm / hostility |

---

### 3. Human vs. LLM Judge Calibration Benchmark (50 Pairs)

- **Exact Agreement**: `96.0%`
- **Agreement within $\pm 1$ Point**: `100.0%`
- **Cohen's Quadratic Weighted Kappa**: `0.9099`
- **Spearman Rank Correlation**: `0.9491` (p = `9.85e-26`)
