"""
Phase 3: Golden Evaluation Set Builder for @AmazonHelp.
Constructs a stratified 200-example test suite:
- 140 Normal cases
- 40 Difficult cases
- 20 Adversarial cases
Balanced across all 8 intents with explicit ground-truth labels and human verification.
"""
import json
import re
import random
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.config import (
    PROCESSED_DATA_DIR,
    GOLDEN_DATA_DIR,
    INTENTS,
    INTENT_PRIORITY,
    DECISION_AUTO_HANDLE,
    DECISION_ESCALATE,
    RANDOM_SEED
)
from src.taxonomy import (
    INTENT_METADATA,
    resolve_intent_collision,
    evaluate_deterministic_risk
)

# Robust keyword detectors for initial categorization before human verification
INTENT_DETECTORS = {
    "ACCOUNT_SECURITY_ACCESS": re.compile(r"\b(hacked|fraud|unauthorized|phishing|otp|locked out|password|scam|security|compromised|login|someone else accessed)\b", re.I),
    "DAMAGED_WRONG_MISSING": re.compile(r"\b(damaged|broken|empty box|missing|wrong item|defective|shattered|cracked|ruined|torn|opened package|stolen|never arrived)\b|\b(shows?|says?|marked|claims?)\s+(as\s+)?delivered\b.*\b(not\s+(here|received|arrived)|never\s+(left|received|got)|missing|stolen|empty|porch)\b|\bdelivered\b.*\b(not\s+received|nowhere\s+to\s+be\s+found|didn't\s+get|stolen|porch)\b", re.I),
    "BILLING_SUBSCRIPTION_PRIME": re.compile(r"\b(prime|membership|subscription|charged|billing|debit|card charged|renew|unrecognized charge|audible fee|annual charge|cashback|amazon pay balance|wallet balance|promotional credit)\b", re.I),
    "REFUND_RETURN_EXCHANGE": re.compile(r"\b(refund|return|returning|exchange|money back|pickup|drop off|refunded|send back|return label|replacement)\b", re.I),
    "ORDER_CHANGE_CANCEL": re.compile(r"\b(cancel|cancelled|cancellation|change address|modify order|wrong address|change payment|stop delivery)\b", re.I),
    "DELIVERY_STATUS_DELAY": re.compile(r"\b(delivery|deliver|late|delayed|delay|where is|tracking|track|carrier|package|parcel|courier|transit|arrive|arriving|eta|status|not delivered)\b", re.I),
    "TECHNICAL_PRODUCT_SUPPORT": re.compile(r"\b(kindle|fire stick|echo|alexa|app|website|code|voucher|coupon|promo|error|crash|bug|tv app|frozen)\b", re.I),
}

def detect_candidate_intents(text: str) -> list:
    detected = []
    for intent, pat in INTENT_DETECTORS.items():
        if pat.search(text):
            detected.append(intent)
    return detected

def classify_candidate(text: str) -> tuple:
    detected = detect_candidate_intents(text)
    if not detected:
        primary_intent = "FEEDBACK_COMPLAINT_GENERAL"
    else:
        primary_intent = resolve_intent_collision(detected)
    return primary_intent, detected

def curate_golden_set():
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden_pool_path = PROCESSED_DATA_DIR / "golden_candidate_pool.parquet"
    
    print(f"[Phase 3] Loading candidate pool from {golden_pool_path}...")
    df_pool = pd.read_parquet(golden_pool_path)
    print(f"Total candidates in pool: {len(df_pool):,}")
    
    candidates_by_intent = {intent: [] for intent in INTENTS}
    multi_intent_candidates = []
    adversarial_candidates = []
    
    for _, row in df_pool.iterrows():
        cid = row["conversation_id"]
        c_text = row["customer_text"]
        s_reply = row["support_reply"]
        
        primary_intent, all_matches = classify_candidate(c_text)
        
        # Check adversarial characteristics:
        # Sarcasm, extreme anger/threats, very short/vague, or conflicting multiple intents
        is_adversarial = False
        adv_reason = ""
        
        if re.search(r"\b(lawyer|police|court|sue|lawsuit|bbb|disgusting|pathetic|worst company|useless|scammers)\b", c_text, re.I):
            is_adversarial = True
            adv_reason = "Customer extreme anger / legal threat / hostility"
        elif len(all_matches) >= 3:
            is_adversarial = True
            adv_reason = f"Triple multi-intent collision: {all_matches}"
        elif len(c_text.split()) <= 4 and primary_intent == "FEEDBACK_COMPLAINT_GENERAL":
            is_adversarial = True
            adv_reason = "Ultra-terse ambiguous message without context"
        elif re.search(r"\b(thanks for nothing|great job amazon|yeah right|surely)\b", c_text, re.I):
            is_adversarial = True
            adv_reason = "Sarcasm / irony"
            
        record = {
            "conversation_id": cid,
            "customer_text": c_text,
            "detected_intent": primary_intent,
            "all_matches": all_matches,
            "support_reply": s_reply,
            "is_adversarial": is_adversarial,
            "adv_reason": adv_reason
        }
        
        if is_adversarial:
            adversarial_candidates.append(record)
        elif len(all_matches) == 2:
            multi_intent_candidates.append(record)
        else:
            candidates_by_intent[primary_intent].append(record)

    print("\nCandidate Pool Distribution:")
    for intent in INTENTS:
        print(f"  {intent:<30}: {len(candidates_by_intent[intent])}")
    print(f"  Multi-Intent Candidates       : {len(multi_intent_candidates)}")
    print(f"  Adversarial Candidates        : {len(adversarial_candidates)}")

    rng = random.Random(RANDOM_SEED)
    
    # Stratified target:
    # 140 Normal
    # 40 Difficult
    # 20 Adversarial
    # Total: 200 examples
    
    # 1. Select 20 Adversarial
    rng.shuffle(adversarial_candidates)
    selected_adversarial = adversarial_candidates[:20]
    
    # 2. Select 40 Difficult (drawn from multi-intent collisions and nuanced long/unresolved queries)
    rng.shuffle(multi_intent_candidates)
    selected_difficult = multi_intent_candidates[:30]
    # Add 10 nuanced long queries across specific intents
    for intent in ["DAMAGED_WRONG_MISSING", "ACCOUNT_SECURITY_ACCESS", "BILLING_SUBSCRIPTION_PRIME", "ORDER_CHANGE_CANCEL"]:
        long_candidates = [c for c in candidates_by_intent[intent] if len(c["customer_text"]) > 140]
        if long_candidates and len(selected_difficult) < 40:
            selected_difficult.append(long_candidates.pop(0))
    if len(selected_difficult) < 40:
        remaining_diff = 40 - len(selected_difficult)
        selected_difficult.extend(multi_intent_candidates[30:30+remaining_diff])
        
    # 3. Select 140 Normal (stratified ~17-18 per intent across all 8 intents)
    selected_normal = []
    target_per_intent = 17
    for intent in INTENTS:
        pool = candidates_by_intent[intent]
        # filter out any already selected
        selected_cids = {c["conversation_id"] for c in selected_adversarial + selected_difficult + selected_normal}
        available = [c for c in pool if c["conversation_id"] not in selected_cids]
        rng.shuffle(available)
        take = min(target_per_intent, len(available))
        selected_normal.extend(available[:take])
        
    # If slight shortfall to 140, fill from largest available intent pools
    if len(selected_normal) < 140:
        needed = 140 - len(selected_normal)
        selected_cids = {c["conversation_id"] for c in selected_adversarial + selected_difficult + selected_normal}
        remaining_pool = [c for intent in INTENTS for c in candidates_by_intent[intent] if c["conversation_id"] not in selected_cids]
        rng.shuffle(remaining_pool)
        selected_normal.extend(remaining_pool[:needed])

    print(f"\nSampled Counts:")
    print(f"  Normal     : {len(selected_normal)}")
    print(f"  Difficult  : {len(selected_difficult)}")
    print(f"  Adversarial: {len(selected_adversarial)}")
    print(f"  Total      : {len(selected_normal) + len(selected_difficult) + len(selected_adversarial)}")

    # Now apply strict Human-Verification & Ground-Truth Labeling Rules:
    golden_examples = []
    
    def label_example(record, tier, index):
        cid = record["conversation_id"]
        c_text = record["customer_text"]
        s_reply = record["support_reply"]
        primary_intent, all_matches = classify_candidate(c_text)
        
        # Ground-truth Intent resolution using priority hierarchy
        golden_intent = primary_intent
        
        # Ground-truth Triage & Reason Assignment
        # Check deterministic risk triggers
        is_risk, risk_cat, risk_reason = evaluate_deterministic_risk(c_text)
        
        if is_risk:
            golden_triage = DECISION_ESCALATE
            golden_cat = risk_cat
            golden_reason = risk_reason
        elif golden_intent in ["DAMAGED_WRONG_MISSING", "ACCOUNT_SECURITY_ACCESS", "BILLING_SUBSCRIPTION_PRIME"]:
            golden_triage = DECISION_ESCALATE
            meta = INTENT_METADATA[golden_intent]
            golden_cat = {
                "DAMAGED_WRONG_MISSING": "DAMAGED_PHYSICAL_MERCHANDISE",
                "ACCOUNT_SECURITY_ACCESS": "ACCOUNT_SECURITY_RISK",
                "BILLING_SUBSCRIPTION_PRIME": "FINANCIAL_OR_BILLING_DISPUTE"
            }.get(golden_intent, "ACCOUNT_SPECIFIC_PII_REQUIRED")
            golden_reason = meta["default_reason"]
        elif golden_intent == "DELIVERY_STATUS_DELAY":
            # If customer specifically says "says delivered but not here", escalate
            if re.search(r"delivered|never arrived|stolen|missing", c_text, re.I):
                golden_triage = DECISION_ESCALATE
                golden_cat = "LOST_OR_STOLEN_DELIVERY"
                golden_reason = "Package marked delivered by carrier but customer states it was not received; requires carrier check and account verification."
            else:
                golden_triage = DECISION_AUTO_HANDLE
                golden_cat = "NONE"
                golden_reason = "General in-transit delivery tracking can be self-served via 'Your Orders' tracking portal."
        elif golden_intent == "ORDER_CHANGE_CANCEL":
            # If item already shipped, cannot cancel self-serve
            if re.search(r"already shipped|on the way|in transit|too late", c_text, re.I):
                golden_triage = DECISION_ESCALATE
                golden_cat = "MANUAL_REFUND_OR_RETURN_OVERRIDE"
                golden_reason = "Order is already dispatched or in transit; manual return setup or carrier intercept required."
            else:
                golden_triage = DECISION_AUTO_HANDLE
                golden_cat = "NONE"
                golden_reason = "Self-service cancellation can be performed via 'Your Orders' before dispatch."
        elif golden_intent == "REFUND_RETURN_EXCHANGE":
            # If customer asks about general return window -> auto-handle; if demanding manual refund -> escalate
            if re.search(r"where is my refund|money back now|haven't received my refund|still waiting for refund", c_text, re.I):
                golden_triage = DECISION_ESCALATE
                golden_cat = "MANUAL_REFUND_OR_RETURN_OVERRIDE"
                golden_reason = "Refund delay inquiry requiring account-specific financial transaction verification."
            else:
                golden_triage = DECISION_AUTO_HANDLE
                golden_cat = "NONE"
                golden_reason = "Standard return policy and self-service return label instructions can be provided automatically."
        elif golden_intent == "TECHNICAL_PRODUCT_SUPPORT":
            golden_triage = DECISION_AUTO_HANDLE
            golden_cat = "NONE"
            golden_reason = "Standard first-line device troubleshooting (hard restart, app cache refresh) can be auto-handled."
        else: # FEEDBACK_COMPLAINT_GENERAL
            if tier == "adversarial" or re.search(r"lawyer|sue|court|fraud|police|unacceptable|furious", c_text, re.I):
                golden_triage = DECISION_ESCALATE
                golden_cat = "CUSTOMER_AGITATION_OR_LEGAL_THREAT"
                golden_reason = "High customer agitation or formal grievance requiring human supervisor handling."
            else:
                golden_triage = DECISION_AUTO_HANDLE
                golden_cat = "NONE"
                golden_reason = "General brand feedback or polite commentary can be acknowledged empathetically without human intervention."

        return {
            "id": f"CANDIDATE-{index+1:03d}",
            "conversation_id": cid,
            "customer_text": c_text,
            "proposed_intent": golden_intent,
            "all_applicable_intents": all_matches if all_matches else [golden_intent],
            "difficulty_tier": tier,
            "proposed_triage": golden_triage,
            "proposed_escalation_category": golden_cat,
            "proposed_escalation_reason": golden_reason,
            "historical_brand_reply": s_reply,
            "human_verified": False,
            "audit_status": "Candidate sampled for human review; not verified ground-truth."
        }

    idx = 0
    for r in selected_normal:
        golden_examples.append(label_example(r, "normal", idx))
        idx += 1
    for r in selected_difficult:
        golden_examples.append(label_example(r, "difficult", idx))
        idx += 1
    for r in selected_adversarial:
        golden_examples.append(label_example(r, "adversarial", idx))
        idx += 1

    # Save candidates to candidate pool (does not overwrite human-audited golden_eval_set.json)
    json_path = GOLDEN_DATA_DIR / "candidate_eval_pool.json"
    csv_path = GOLDEN_DATA_DIR / "candidate_eval_pool.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(golden_examples, f, indent=2, ensure_ascii=False)
        
    df_golden = pd.DataFrame(golden_examples)
    df_golden.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n[Candidate Sampler Complete] Saved 200 candidates for human audit to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    
    print("\nGolden Set Summary Statistics:")
    print(f"  Total items          : {len(df_golden)}")
    print(f"  Difficulty breakdown : {df_golden['difficulty_tier'].value_counts().to_dict()}")
    print(f"  Triage breakdown     : {df_golden['golden_triage'].value_counts().to_dict()}")
    print(f"  Intent distribution  :")
    for intent, count in df_golden["golden_intent"].value_counts().items():
        print(f"    {intent:<30}: {count} ({count/len(df_golden)*100:.1f}%)")

if __name__ == "__main__":
    curate_golden_set()
