# System Evaluation Report: AI Support Agent for @AmazonHelp

**Candidate**: Hiver SDE Intern Applicant  
**Brand Evaluated**: `@AmazonHelp` (E-Commerce & Digital Logistics)  
**System Status**: Evaluation-First Support-Agent Prototype (Production-Minded)  
**Dataset Vintage**: ThoughtVector Customer Support on Twitter (~794k multi-turn threads; 81,092 @AmazonHelp conversations)  
**Core Thesis**: *"The objective is not maximum automation. The objective is maximum safe resolution."*  

---

## 1. Executive Summary & Problem Framing

Customer support on public social channels presents a severe tension between **automation efficiency** and **operational risk**. While automated canned responses decrease First Response Time (FRT), an inaccurate resolution—such as misdiagnosing a stolen package as a routine transit delay or auto-resolving an account takeover inquiry—inflicts catastrophic brand damage, financial chargebacks, and customer churn.

### What "Good" Means for @AmazonHelp
For `@AmazonHelp`, an AI support prototype is deemed trustworthy only if it adheres to four non-negotiable operational principles:
1. **Safety-First Triage Over Automation**: An auto-handled interaction must be demonstrably safe. When ambiguity exists, the system must escalate to a human specialist.
2. **Zero Public PII Disclosure**: Customer account IDs, credit card numbers, email addresses, and physical locations must *never* be requested or handled on a public Twitter thread. Escalated conversations must route to authenticated Direct Message (DM) channels or official contact portals.
3. **Empathetic, Brand-Aligned De-escalation**: Tone must mirror Amazon's historical resolution standard: polite acknowledgment, clear guidance, and agent initial sign-offs (`^CS`).
4. **Historical Grounding Without Policy Hallucination**: Replies must reflect documented historical brand behaviors rather than inventing hypothetical service level agreements (SLAs) or synthetic tracking links.

### Scope Boundary: What We Deliberately Chose NOT to Build
To preserve engineering integrity and prevent safety regressions, we explicitly excluded:
- **Autonomous Financial Authorizations**: The prototype does *not* execute automated refunds or account credits without human supervisory sign-off.
- **Speculative Policy/SLA Guarantees**: We deliberately avoided hardcoding arbitrary corporate rules (e.g. claiming a rigid ">48h delivery delay SLA" or asserting current return windows) because historical Twitter threads cannot be conflated with live internal Amazon operating policies.
- **Unconstrained Free-Form Generation**: We do not allow the model to generate arbitrary external URLs. All link guidance is restricted to safe, verified placeholder tokens (`[link]`) representing official Amazon authenticated routes (`Your Orders`, `Contact Us`).

---

## 2. Empirical Brand Selection & Intent Discovery

### Empirical Multi-Brand Audit
Rather than selecting `@AmazonHelp` based solely on raw tweet volume, we scored candidate brands across an objective 6-factor decision matrix:

$$\text{Score} = 0.25 C_{\text{usable}} + 0.20 R_{\text{resol}} + 0.20 D_{\text{vocab}} + 0.15 T_{\text{thread}} + 0.10 Q_{\text{quality}} + 0.10 E_{\text{eval}}$$

| Candidate Brand | Vertical | Total Convs | Actionable Resol % | Vocab Entropy | Multi-Turn % ($\ge 3$ turns) | Data Quality Score | Eval Difficulty Score | **Composite Score** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`@AmazonHelp`** | **E-Commerce / Logistics** | **81,092** | **42.4%** | **10.30** | **60.1%** | **98.5** | **51.8** | **83.52 (Winner)** |
| **`@AppleSupport`** | Consumer Tech / OS | 76,639 | 53.6% | 9.47 | 33.0% | 99.1 | 42.5 | **70.75** |
| **`@Uber_Support`** | Mobility / Gig Economy | 41,185 | 25.6% | 9.52 | 33.7% | 98.6 | 75.2 | **67.65** |
| **`@Delta`** | Airlines / Travel | 25,151 | 27.0% | 9.72 | 39.0% | 98.3 | 42.8 | **66.65** |
| **`@SpotifyCares`** | Digital Streaming | 27,910 | 29.3% | 9.53 | 36.6% | 98.2 | 44.8 | **65.17** |
| **`@TMobileHelp`** | Telecom / Carrier | 22,322 | 27.1% | 9.64 | 34.7% | 97.5 | 44.6 | **64.78** |

`@AmazonHelp` achieved the highest composite score due to its high multi-turn conversation richness ($60.1\%$), superior vocabulary diversity ($10.30$ bits), and high-stakes triage balance between routine tracking FAQs and severe escalations (stolen packages, account lockouts, payment disputes).

### The 8-Intent Taxonomy & Multi-Intent Hierarchy
Auditing 1,000 uncurated customer messages revealed that general complaints and unspecific feedback accounted for $\approx 38\%$ of incoming volume. To prevent class imbalance from distorting evaluation metrics, we formalized an 8-intent MECE taxonomy with explicit boundary rules:

1. `DELIVERY_STATUS_DELAY`: Inquiries regarding transit status, tracking numbers, or carrier delays. *(Boundary: delivered-but-missing inquiries are excluded and classified under `DAMAGED_WRONG_MISSING`)*.
2. `DAMAGED_WRONG_MISSING`: Physical package defects, wrong items delivered, empty boxes, or missing packages marked delivered (porch pirate).
3. `REFUND_RETURN_EXCHANGE`: Return window policy, return shipping labels, or replacement requests.
4. `ORDER_CHANGE_CANCEL`: Order cancellation or shipping address updates pre-dispatch.
5. `BILLING_SUBSCRIPTION_PRIME`: Unrecognized charges, Prime membership renewals, digital wallet cashback, or subscription fee disputes.
6. `ACCOUNT_SECURITY_ACCESS`: Compromised accounts, 2FA/OTP failures, password lockouts, or phishing reports.
7. `TECHNICAL_PRODUCT_SUPPORT`: Hardware troubleshooting (Kindle, Echo, Fire TV) and app/digital glitches.
8. `FEEDBACK_COMPLAINT_GENERAL`: Sarcasm, praise/shoutouts, brand rants, or commentary lacking order-specific identifiers.

#### Multi-Intent Priority Hierarchy
When customer messages span multiple categories (which occurs in $\approx 9.7\%$ of cases), the system resolves collisions via a strict risk-dominant hierarchy:

$$\text{Security/Fraud} \succ \text{Damaged/Missing} \succ \text{Billing Dispute} \succ \text{Refund/Return} \succ \text{Order Change} \succ \text{Delivery} \succ \text{Technical} \succ \text{General}$$

---

## 3. System Architecture & Safety-First Triage Engine

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
            (Deterministic Safety Rules
            Cannot Be Overridden by LLM)
                       │
              ┌────────┴────────┐
              ▼                 ▼
         AUTO_HANDLE         ESCALATE
              │                 │
               └────────┬────────┘
                        ▼
                 REPLY GENERATOR
       (Retrieval-Conditioned Synthesis)
                        │
                        ▼
               RESPONSE SANITIZER
              (PII & Channel Guard)
                        │
                        ▼
                   FINAL JSON
```

### Deterministic Safety Guardrails & Retrieval-Conditioned Reply Synthesis
1. **Deterministic Safety Engine**: A core architectural principle of our system is that **the model cannot override deterministic safety rules**. If regex/pattern engines detect high-risk signals (e.g. *"someone hacked my account"*, *"police"*, *"lawyer"*, *"unauthorized charge"*, *"marked delivered but never arrived"*), the triage engine unconditionally forces an `ESCALATE` decision with a structured justification.
2. **Retrieval-Conditioned Canonical Reply Synthesis**: Rather than blindly replaying raw historical tweets verbatim (which risks dead links, stale 2017 policies, and privacy violations as demonstrated by Baseline 1's 8.5% pass rate), or relying on an unconstrained generative LLM that can hallucinate corporate return policies, our agent uses **Retrieval-Conditioned Canonical Synthesis**:
   - Queries the 55,011-record historical knowledge base for top matching historical resolution pairs (`evidence[0]`).
   - Extracts authentic historical agent voice tags (`^CS`, `^GR`, etc.) and channel cues.
   - Populates a policy-safe, privacy-sanitized draft reply with guaranteed deep-link anchors (`[link]`) and active regex PII sanitization.

---

## 4. Headline Results vs. Dual Baselines

All three systems were evaluated on an independently hand-verified **Golden Evaluation Suite of 200 stratified cases** (140 Normal, 40 Difficult, 20 Adversarial), strictly held out from the 55,011-case historical knowledge base (**zero conversation ID overlap, zero customer text leakage**).

### Headline Benchmark Comparison

| Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Real-World Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Safe Auto-Handle Precision** | `0.4800` | `0.5429` | **`0.9048`** | **Hero Metric**: when choosing Auto-Handle, is it truly safe? |
| **Escalation Recall** | `0.0000` | `0.3846` | **`0.9038`** | **Hero Metric**: coverage of critical security and financial risks |
| **Missed Escalation Rate** | `1.0000` | `0.6154` | **`0.0962`** | **Critical Safety Failure**: true risks erroneously automated |
| **False Escalation Rate** | `0.0000` | `0.2083` | **`0.0104`** | Over-escalation rate (human agent queue bloat & cost) |
| **Intent Macro-F1** | `0.0227` | `0.9039` | **`0.9672`** | Unskewed multi-class classification metric |
| **Intent Overall Accuracy** | `0.1000` | `0.9000` | **`0.9700`** | Classification correctness across all 8 intents |
| **Escalation Precision** | `0.0000` | `0.6667` | **`0.9895`** | Proportion of escalated queries that legitimately require humans |
| **Grounded Reply Pass Rate** | `0.0000` | `0.0850` | **`0.8650`** | Deterministic 4-Criteria Rubric Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `1.0000` | `1.0000` | **`1.0000`** | 100% compliance with deterministic PII-safety rules (zero credential solicitation) |
| **ROUGE-L Diagnostic** | `0.1053` | `0.1514` | **`0.1029`** | Lexical overlap against historical 2017 tweets |

### Performance Breakdown Across Difficulty Tiers (Macro-F1)

| Difficulty Tier | Sample Count | Baseline 0 | Baseline 1 | Proposed AI Agent | Performance Drop (Adversarial vs Normal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Cases** | 140 | `0.0256` | `0.9389` | **`0.9733`** | Clean single-intent queries |
| **Difficult Cases** | 40 | `0.0081` | `0.8306` | **`1.0000`** | Nuanced multi-intent queries |
| **Adversarial Cases** | 20 | `0.0435` | `0.7139` | **`0.7891`** | Material drop; qualitative review highlights sarcasm, ambiguity, hostility |

---

## 5. Automated Multi-Criteria Quality Rubric & Reply Evaluation

Rather than relying purely on superficial n-gram overlap metrics (such as BLEU or ROUGE, which heavily penalize valid lexical variation), reply generation is evaluated using an explicit, deterministic multi-criteria quality rubric implemented in `src/llm_judge.py` (`ReplyQualityRubric`).

### Evaluation Dimensions (1–5 Scale)
1. **Groundedness & Factual Realism (1–5)**:
   Assesses whether the response accurately incorporates resolution concepts matching the customer's specific problem category (e.g., dispatch tracking for delays, prepaid return labels for refunds, carrier tracing for stolen packages).
2. **Brand Voice & Empathy (1–5)**:
   Measures courteous, brand-aligned sign-offs (e.g., `^CS`, `^GR`) and empathetic acknowledgment (*"Sorry to hear"*, *"We want to make this right"*), matching `@AmazonHelp` historical norms.
3. **Actionability & Resolution Guidance (1–5)**:
   Measures whether the customer is provided an unambiguous, concrete next step (e.g., self-service account portal link `[link]` for routine tasks, or a secure private direct message handoff for escalations).
4. **Channel & PII Safety (1–5)**:
   Strict zero-tolerance gate: penalizes public solicitation of passwords, full credit card numbers, CVVs, or sensitive credentials on public social channels.

### Pass/Fail Criteria
A response is scored as a **Pass** if:
- `channel_safety == 5` (zero credential solicitation)
- `groundedness >= 4`
- `actionability >= 4`
- `overall_score >= 3.8`

Under this rubric:
- **Baseline 0 (Trivial Constant Reply)**: `0.0%` pass rate (lacks intent-specific grounding and actionability).
- **Baseline 1 (Simple 1-NN Retrieval)**: `8.5%` pass rate (historical raw tweets frequently contain broken links, incomplete context, or missing empathy markers).
- **Proposed AI Agent**: **`86.5%`** pass rate (structured, grounded response generation with dynamic brand voice and verified self-service/escalation paths).

### Note on Human Calibration in this Prototype
> [!NOTE]
> To maintain strict methodological transparency, we explicitly note that an independent multi-annotator human calibration study (e.g., Cohen's Kappa or Spearman correlation against rater panels) was **not conducted** for this prototype. The rubric serves as an automated, transparent, rule-based regression test suite. Conducting an extensive human rater calibration study and evaluating against fine-tuned judge models is scoped as a priority for production deployment.

---

## 6. Failure Analysis: Top 5 Real Failure Modes

### Failure Mode 1: Positive Sentiment False Escalation
- **Customer Tweet (`GOLDEN-008`)**: *"Shoutout to sherry from @AmazonHelp for helping me with me lost package all my family’s Christmas presents..."*
- **Ground Truth**: `AUTO_HANDLE` (Appreciation acknowledgment under `FEEDBACK_COMPLAINT_GENERAL`)
- **Agent Prediction**: `ESCALATE` (Category: `LOST_OR_STOLEN_DELIVERY`)
- **Root Cause Hypothesis**: The deterministic keyword detector matched `"lost package"` in a retrospective praise context, failing to recognize positive sentiment.
- **Remediation**: Add a syntactic dependency parser to differentiate past-resolved praise clauses (*"helped me with"*) from active operational grievances.

### Failure Mode 2: Uncaptured Promotional / Wallet Credit Disputes
- **Customer Tweet (`GOLDEN-185`)**: *"Another pathetic experience from Amazon india.not received my amazon pay cashback since 2 days..."*
- **Ground Truth**: `ESCALATE` (Financial dispute under `BILLING_SUBSCRIPTION_PRIME`)
- **Agent Prediction**: `AUTO_HANDLE` (Category: `FEEDBACK_COMPLAINT_GENERAL`)
- **Root Cause Hypothesis**: Billing patterns captured credit cards and subscriptions, but initially missed digital wallet cashbacks (`"amazon pay cashback"`), falling back to general complaint.
- **Remediation**: Expanded `BILLING_SUBSCRIPTION_PRIME` to encompass digital stored-value balances and promotional bank cashbacks.

### Failure Mode 3: Implicit Multiple-Touchpoint Agitation
- **Customer Tweet (`GOLDEN-183`)**: *"Standard copy paste answers without even thinking about what the issue is and why customer is reaching out..."*
- **Ground Truth**: `ESCALATE` (Customer agitation requiring senior human intervention)
- **Agent Prediction**: `AUTO_HANDLE` (Generic empathetic acknowledgment)
- **Root Cause Hypothesis**: The customer did not use explicit escalation keywords, but was furious about receiving repetitive bot replies. The system auto-handled the query with another bot reply.
- **Remediation**: Implement a meta-complaint detector that flags customer grievances regarding automated or canned support itself.

### Failure Mode 4: Instant Bank Discount Collisions
- **Customer Tweet (`GOLDEN-193`)**: *"after buying the OnePlus 5T I didn't get any instant discount neither any cashback."*
- **Ground Truth**: `ESCALATE` (Post-purchase checkout payment override)
- **Agent Prediction**: `AUTO_HANDLE`
- **Root Cause Hypothesis**: Ambiguity between pre-purchase promo code technical bugs (auto-handleable) and post-purchase missing financial credits (escalate).
- **Remediation**: Any transaction where payment has already cleared must default to financial escalation.

### Failure Mode 5: Grounded Historical URL Link Rot
- **Observed Behavior**: Historical 2017 tweets contain defunct `t.co` shortlinks and regional URL paths that no longer resolve.
- **Remediation**: All links in our agent are strictly mapped to dynamic, tokenized deep-link anchors (`[link]`) resolved at runtime by the host environment.

---

## 7. Mandatory Section: "What is Misleading About My Headline Number?"

A headline metric of **`0.9038` Escalation Recall** and **`0.9048` Safe Auto-Handle Precision** (with an exceptional **`0.9895` Escalation Precision**) is strong, but presenting it without critical qualification would be intellectually dishonest:

1. **Adversarial Tier Performance Drop**:
   The adversarial tier has materially lower Macro-F1 than the normal tier ($0.7891$ vs. $0.9733$). Qualitative failure analysis suggests sarcasm, hostility, retrospective praise, and ambiguous complaint phrasing as contributing factors across these 20 edge cases, rather than a single causal mechanism.
2. **Stratified Golden Set vs. In-The-Wild Distributional Shift**:
   In our raw 1,000-message discovery audit, $38\%$ of inbound tweets were unstructured rants or praise (`FEEDBACK_COMPLAINT_GENERAL`). Our 200-item golden set intentionally capped this class at $13.5\%$ to test discriminative competence. In live production, the raw stream contains far higher conversational noise.
3. **Asymmetry of Triage Costs**:
   A $9.62\%$ Missed Escalation Rate sounds modest, but in customer service, **errors are not symmetric**. Erroneously auto-handling a single customer whose package was stolen or whose account was compromised can trigger credit card chargebacks, formal regulatory complaints, and churn. In a 100,000-ticket/day queue, a $9.6\%$ missed escalation rate represents nearly 10,000 customer failure points daily.
4. **Intent Correctness Does Not Equal Problem Resolution**:
   Classifying a tweet correctly as `DELIVERY_STATUS_DELAY` does not mean the customer was satisfied. If the carrier lost the shipment, sending a generic tracking link merely delays customer frustration.
5. **Historical Dataset Vintage (2017 vs. Present)**:
   The TWCS dataset reflects Twitter policies and operational workflows from 2017. Modern customer service relies on rich in-app messaging, automated authentication handoffs, and updated returns portals that did not exist when these tweets were authored.
6. **Automated Rubric vs. Nuanced Human Judgment**:
   Our reply evaluation relies on a transparent, deterministic multi-criteria rubric (`ReplyQualityRubric`) rather than human panels or opaque LLM judge APIs. While deterministic rubrics provide 100% reproducible and fast regression checks, they cannot detect subtle conversational nuances, natural human empathy variation, or brand tone subtleties as effectively as calibrated human review panels.

---

## 8. What We Would Build Next with One More Week

1. **Multi-Turn Session State Tracking**:
   Extend the agent from single-turn tweet processing to dynamic thread graph tracking, maintaining conversation memory across turns to prevent re-sending links the customer already tried.
2. **Production Order API Tool-Calling**:
   Equip the agent with structured mock tools (`lookup_order_status(order_id)`, `check_delivery_sla(order_id)`) to inspect actual simulated logistics data before making an escalation decision.
3. **Active Learning & Uncertainty Triage**:
   Implement conformal prediction to isolate queries where intent confidence falls between $0.40$ and $0.65$, routing them to a secondary human-in-the-loop review queue for continuous model fine-tuning.
4. **Sub-50ms Edge Inference**:
   Distill the classification and triage logic into a quantized small language model (SLM) or optimized ONNX runtime to reduce inference latency from hundreds of milliseconds to under 30ms.
