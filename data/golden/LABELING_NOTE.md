# Golden Evaluation Suite: Sampling & Human Annotation Note

**Author / Annotator**: Sanskar Sinha  
**Target Brand**: `@AmazonHelp`  
**Dataset Size**: 200 Hand-Labelled Stratified Conversations  
**Verification Status**: 100% Manually Audited & Verified  

---

## 1. Sampling Methodology & Distribution Strategy

The 200 golden examples were sampled from a strictly held-out candidate pool of 1,200 `@AmazonHelp` conversations (disjoint by conversation ID from the 55,011-record historical knowledge base):

1. **Stratification Across 8 Intents**:
   To prevent class imbalance from distorting metrics (where general feedback represented ~38% of raw Twitter volume), we established strict category quotas:
   - `BILLING_SUBSCRIPTION_PRIME`: 45 cases (22.5%)
   - `DAMAGED_WRONG_MISSING`: 27 cases (13.5%)
   - `FEEDBACK_COMPLAINT_GENERAL`: 27 cases (13.5%)
   - `REFUND_RETURN_EXCHANGE`: 22 cases (11.0%)
   - `ACCOUNT_SECURITY_ACCESS`: 21 cases (10.5%)
   - `DELIVERY_STATUS_DELAY`: 20 cases (10.0%)
   - `ORDER_CHANGE_CANCEL`: 19 cases (9.5%)
   - `TECHNICAL_PRODUCT_SUPPORT`: 19 cases (9.5%)

2. **Stratification by Difficulty Tier**:
   - **Normal Cases (140)**: Clear single-intent customer inquiries (e.g. routine in-transit tracking, return window policy questions, Kindle restart steps).
   - **Difficult Cases (40)**: Multi-sentence inquiries, multi-intent collisions (e.g. delayed delivery + refund demand), and missing order context.
   - **Adversarial Cases (20)**: Sarcasm, extreme customer hostility/legal threats, carrier property damage, and ambiguous fragments.

3. **Balanced Triage Boundary**:
   - **`ESCALATE`**: 104 cases (52.0%)
   - **`AUTO_HANDLE`**: 96 cases (48.0%)
   This near-50/50 balance ensures that triage accuracy and safe auto-handle precision cannot be achieved through majority-class guessing.

---

## 2. Human Annotation Protocol & Ground-Truth Rules

Every example was individually reviewed and validated against Amazon's operational support boundaries:

- **Rule 1 (Theft vs. Delay)**: Any tweet stating a package is marked delivered but was not found (e.g. porch theft) is strictly classified as `DAMAGED_WRONG_MISSING` and `ESCALATE`, never `DELIVERY_STATUS_DELAY`.
- **Rule 2 (Praise vs. Grievance)**: Customer shoutouts thanking agents for finding previously lost items are classified as `FEEDBACK_COMPLAINT_GENERAL` and `AUTO_HANDLE`, preventing false keyword-driven escalation.
- **Rule 3 (Financial & Account Invariance)**: Inquiries regarding blocked accounts, unauthorized debits, and promotional cashback disputes unconditionally require human escalation (`ESCALATE`) to protect customer PII.
- **Rule 4 (Driver Property Damage & Hostility)**: Severe courier misconduct (e.g. damaged mailboxes, broken locks) or formal legal threats (`"lawyer"`, `"consumer court"`) are escalated immediately under `CUSTOMER_AGITATION_OR_LEGAL_THREAT`.

---

## 3. Data Integrity & Leakage Verification

- **Conversation ID Overlap**: Exactly **0** overlapping conversation IDs between Golden Set and Historical KB.
- **Customer Text Overlap**: Exactly **0** identical customer query strings.
- **Mean Top-1 Retrieval Cosine Similarity**: **0.3718** (No memorization).
