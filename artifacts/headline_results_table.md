# Benchmark Results: @AmazonHelp AI Support Agent vs. Dual Baselines

**Evaluation Suite**: 200 Stratified Hand-Labelled Cases (140 Normal, 40 Difficult, 20 Adversarial)  
**Historical Evidence Base**: 55,011 disjoint @AmazonHelp customer->brand resolution pairs  

---

### 1. Headline System Comparison

| Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Metric Priority / Safety Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Macro-F1** | `0.0227` | `0.9039` | **`0.8441`** | **Primary Intent Metric** (unskewed by class imbalance) |
| **Intent Overall Accuracy** | `0.1000` | `0.9000` | **`0.8700`** | Overall classification correctness |
| **Safe Auto-Handle Precision** | `0.4800` | `0.5429` | **`0.8947`** | **Primary Safety Metric** (when auto-handling, is it truly safe?) |
| **Missed Escalation Rate** | `1.0000` | `0.6154` | **`0.0962`** | **Critical Hazard** (true risk queries dangerously auto-handled) |
| **False Escalation Rate** | `0.0000` | `0.2083` | **`0.1146`** | Over-escalation rate (queue cost / human agent burden) |
| **Escalation Recall** | `0.0000` | `0.3846` | **`0.9038`** | Coverage of critical security/financial risks |
| **Escalation Precision** | `0.0000` | `0.6667` | **`0.8952`** | Cleanliness of human triage queue |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.9000`** | Deterministic 4-Criteria Rubric Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | Deterministic rule compliance (zero credential solicitation) |
| **ROUGE-L Similarity** | `0.1053` | `0.1514` | **`0.1014`** | Lexical alignment with historical Amazon resolutions |

---

### 2. Difficulty Tier Performance Breakdown (Macro-F1)

| Difficulty Tier | Sample Count | Baseline 0 | Baseline 1 | Proposed AI Agent | Performance Drop (Adversarial vs Normal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Cases** | 140 | `0.0256` | `0.9389` | **`0.8444`** | Baseline reference |
| **Difficult Cases** | 40 | `0.0081` | `0.8306` | **`0.8889`** | Nuanced multi-intent queries |
| **Adversarial Cases** | 20 | `0.0435` | `0.7139` | **`0.8691`** | High-friction / sarcasm / hostility |
