# Engineering Decision Log: AI Support Agent for @AmazonHelp

This document records the **16 non-obvious engineering decisions** made during the design, implementation, and evaluation of the `@AmazonHelp` AI Customer Support Agent.

---

### 1. Brand Selection via Multi-Factor Matrix Rather Than Raw Volume
- **Decision**: Audited 6 top candidate brands across 6 weighted dimensions (`AmazonHelp`, `AppleSupport`, `Uber_Support`, `Delta`, `SpotifyCares`, `TMobileHelp`) rather than defaulting to `AmazonHelp` solely because of its tweet count.
- **Why**: High raw volume can mask low conversational richness (e.g. `Uber_Support` had 41k conversations but 74% were canned bot redirects). `AmazonHelp` legitimately won on multi-turn depth (60.1%) and vocabulary entropy (10.30), providing a defensible engineering choice.

### 2. Discarding the Raw Cleaned Text Parquet in Favor of Multi-Turn Conversation Parquet
- **Decision**: Scrapped the initial 185 MB `gorkemsevinc` parquet file upon discovering it was stripped of metadata, and acquired the full 207 MB multi-turn conversation dataset (`TNE-AI`).
- **Why**: Customer support is fundamentally conversational. Single-turn stemmed text destroys thread-level context, parent-child response links, and speaker attribution.

### 3. Deliberate Deprecation of the ">48h SLA" Hardcoded Escalation Rule
- **Decision**: Removed speculative corporate SLA rules (e.g. "escalate if delayed > 48h") from the taxonomy.
- **Why**: Historical 2017 Twitter data cannot establish Amazon's current internal SLAs. Claiming a specific corporate policy without authoritative internal documentation introduces hallucinated operational claims into the evaluation harness.

### 4. Bounding `FEEDBACK_COMPLAINT_GENERAL` in the Golden Evaluation Suite
- **Decision**: Capped general brand complaints at 13.5% (27/200) in the golden set despite them representing ~38% of in-the-wild discovery volume.
- **Why**: Allowing general complaints to dominate 38% of the test suite would artificially inflate accuracy through majority-class guessing. Stratifying the test suite across all 8 intents ensures true discriminative capability is tested.

### 5. Prioritizing Macro-F1 Over Raw Classification Accuracy
- **Decision**: Selected Macro-F1 as the primary intent evaluation metric.
- **Why**: Accuracy is dangerously misleading under class imbalance. Macro-F1 treats critical low-frequency classes (like `ACCOUNT_SECURITY_ACCESS` at ~3%) with equal weight to high-frequency tracking inquiries.

### 6. Strict Disjoint Partitioning Between Golden Test Set and Retrieval KB
- **Decision**: Enforced an explicit programmatic assertion that zero `conversation_id`s in the Golden Evaluation Set exist within the 55,011-record historical knowledge base.
- **Why**: Prevents retrieval data leakage where the RAG pipeline simply looks up the exact tweet it is being evaluated on.

### 7. Deterministic Safety Engine Over Unconstrained LLM Triage
- **Decision**: Made the triage decision engine deterministic and rule-dominant, ensuring the LLM cannot override high-risk triggers (fraud, lost delivered packages, legal threats).
- **Why**: Probabilistic LLM outputs can fail unpredictably on edge cases. When financial loss, theft, or account compromise is present, safety boundaries must be deterministic and verifiable.

### 8. Replacing Raw Historical Shortlinks with Dynamic Link Placeholders (`[link]`)
- **Decision**: Sanitized all historical `t.co` URLs into standardized `[link]` tokens during preprocessing and reply generation.
- **Why**: Historical URLs from 2017 suffer from severe link rot. Direct generation of dead URLs degrades user trust and presents a phishing/security hazard.

### 9. Treating Historical Conversations as Empirical Evidence Rather Than Policy
- **Decision**: Explicitly framed retrieved historical cases as *evidence of historical resolution behavior*, not official current Amazon policy.
- **Why**: Customer service policies change over time. Grounding an agent in past agent behavior is valuable for voice and empathy, but conflating it with live policy leads to policy drift.

### 10. Multi-Intent Collision Priority Hierarchy
- **Decision**: Implemented an explicit priority order (`Security > Damaged/Missing > Billing > Refund > Delivery > Technical > General`) for multi-intent tweets.
- **Why**: When a customer writes *"Package arrived broken and I want my money back"*, assigning the intent to `DAMAGED_WRONG_MISSING` takes precedence because physical item inspection and replacement authorization dictate the operational workflow, not just the refund keyword.

### 11. Core Operational Metric: Safe Auto-Handle Precision Over Automation Rate
- **Decision**: Evaluated the triage engine primarily on *Safe Auto-Handle Precision* ($94.95\%$) and *Missed Escalation Rate* ($4.81\%$) rather than gross auto-handle percentage.
- **Why**: An agent that auto-handles 90% of tickets with 80% safety is an operational disaster. An agent that auto-handles 50% of tickets with 95% safety is immediately deployable.

### 12. Human-in-the-Loop Ground-Truth Verification
- **Decision**: Hand-verified all 200 golden evaluation items rather than accepting pure LLM pseudo-labels.
- **Why**: LLMs have known systematic failure modes on sarcasm, typos, and adversarial prompts. Golden evaluation sets must reflect human ground truth to be scientifically valid.

### 13. Calibration of LLM Judge with Quadratic Weighted Cohen's Kappa
- **Decision**: Validated the automated LLM judge against 50 hand-annotated human ratings, reporting Quadratic Weighted Kappa ($\kappa = 0.9099$), Spearman correlation ($0.9491$), and explicit qualitative disagreement cases.
- **Why**: Simply asserting that an LLM judge was used is insufficient. Proving human-judge alignment provides the scientific defensibility required by Hiver.

### 14. 100% Offline Self-Contained Reproduction Pipeline
- **Decision**: Engineered the pipeline to run fully locally in under 1 minute without mandatory external API keys.
- **Why**: Evaluators running code live must not encounter API key failures, network timeouts, or rate-limiting errors. The entire evaluation harness is self-contained and reproducible.

### 15. Elimination of Circular Rule-Label Leakage via Hand-Verification
- **Decision**: Audited and separated the agent's inference engine from the golden set ground-truth generation. Re-evaluated every single case manually to establish true independent ground truth, correcting delivered-but-missing items, praise false-positives, and wallet disputes (documented in `data/golden/LABELING_NOTE.md`).
- **Why**: Evaluating an agent on data labeled by its own heuristic rules produces an artificial 1.0000 Macro-F1 illusion. Genuine independent hand-verification revealed our true, defensible Macro-F1 of 0.9725, while exposing a materially lower Macro-F1 of 0.7891 on the 20-case adversarial tier.

### 16. Retrieval-Conditioned Canonical Reply Synthesis vs. Raw Text Replay vs. Unconstrained Generative LLM
- **Decision**: Implemented reply drafting via *Retrieval-Conditioned Canonical Synthesis* rather than either raw 1-NN text replay (as in Baseline 1) or an unconstrained generative LLM prompt. The system retrieves historical resolution evidence from 55,011 cases to extract historical agent voice tags (`^CS`, `^GR`, etc.), resolution channel cues, and relevance context, but populates a deterministic, policy-safe response with guaranteed link anchors (`[link]`) and active privacy sanitization.
- **Why**: As proven by Baseline 1, raw historical tweet retrieval achieves a miserable **8.5% pass rate** on the LLM judge because historical tweets contain dead 2017 links (`t.co`), stale policies, and lack required privacy warnings. Conversely, an unconstrained generative LLM introduces severe risks of hallucinating specific corporate guarantees (such as promising 'free replacements' or quoting specific internal warranty terms). Retrieval-conditioned canonical synthesis grounds the agent in historical evidence while maintaining strict, auditable enterprise safety boundaries.
