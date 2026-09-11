# Benchmark Results: @AmazonHelp AI Support Agent vs. Dual Baselines

**Evaluation Suite**: 200 Stratified Hand-Labelled Cases (140 Normal, 40 Difficult, 20 Adversarial)  
**Historical Evidence Base**: 55,011 disjoint @AmazonHelp customer->brand resolution pairs  

---

### 1. Headline System Comparison

| Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Metric Priority / Safety Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Macro-F1** | `0.0238` | `0.9142` | **`0.9834`** | **Primary Intent Metric** (unskewed by class imbalance) |
| **Intent Overall Accuracy** | `0.1050` | `0.9100` | **`0.9850`** | Overall classification correctness |
| **Safe Auto-Handle Precision** | `0.4950` | `0.5571` | **`0.9798`** | **Primary Safety Metric** (when auto-handling, is it truly safe?) |
| **Missed Escalation Rate** | `1.0000` | `0.6139` | **`0.0198`** | **Critical Hazard** (true risk queries dangerously auto-handled) |
| **False Escalation Rate** | `0.0000` | `0.2121` | **`0.0202`** | Over-escalation rate (queue cost / human agent burden) |
| **Escalation Recall** | `0.0000` | `0.3861` | **`0.9802`** | Coverage of critical security/financial risks |
| **Escalation Precision** | `0.0000` | `0.6500` | **`0.9802`** | Cleanliness of human triage queue |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.7650`** | Multi-criteria LLM Judge Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | Zero-leakage compliance (protecting customer identity) |
| **ROUGE-L Similarity** | `0.1053` | `0.1514` | **`0.1037`** | Lexical alignment with historical Amazon resolutions |

---

### 2. Difficulty Tier Performance Breakdown (Macro-F1)

| Difficulty Tier | Sample Count | Baseline 0 | Baseline 1 | Proposed AI Agent | Performance Drop (Adversarial vs Normal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Cases** | 140 | `0.0256` | `0.9454` | **`0.9860`** | Baseline reference |
| **Difficult Cases** | 40 | `0.0081` | `0.8306` | **`1.0000`** | Nuanced multi-intent queries |
| **Adversarial Cases** | 20 | `0.0556` | `0.7481` | **`0.8286`** | High-friction / sarcasm / hostility |

---

### 3. Human vs. LLM Judge Calibration Benchmark (50 Pairs)

- **Exact Agreement**: `96.0%`
- **Agreement within $\pm 1$ Point**: `100.0%`
- **Cohen's Quadratic Weighted Kappa**: `0.9099`
- **Spearman Rank Correlation**: `0.9491` (p = `9.85e-26`)
