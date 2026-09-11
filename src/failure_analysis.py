"""
Phase 9: Failure Analysis Engine for @AmazonHelp.
Identifies and categorizes the top failure modes across intent classification, triage safety, and reply grounding.
"""
import json
import pandas as pd
from pathlib import Path
from src.config import GOLDEN_DATA_DIR, ARTIFACTS_DIR
from src.agent import AmazonSupportAgent

def run_failure_analysis():
    golden_df = pd.read_json(GOLDEN_DATA_DIR / "golden_eval_set.json")
    agent = AmazonSupportAgent()
    
    print("[Failure Analysis] Evaluating all 200 cases to isolate edge failures...")
    failures = {
        "missed_escalations": [],
        "false_escalations": [],
        "multi_intent_borderline": [],
        "adversarial_edge_cases": [],
        "reply_quality_failures": []
    }
    
    from src.llm_judge import ReplyQualityJudge
    judge = ReplyQualityJudge()

    for _, row in golden_df.iterrows():
        pred = agent.process(row["customer_text"])
        j_eval = judge.evaluate_reply(row["customer_text"], pred["reply"], pred["intent"], pred["decision"])
        
        gold_triage = row["golden_triage"]
        pred_triage = pred["decision"]
        gold_intent = row["golden_intent"]
        pred_intent = pred["intent"]
        tier = row["difficulty_tier"]
        
        record = {
            "id": row["id"],
            "tier": tier,
            "customer_text": row["customer_text"],
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
            "gold_triage": gold_triage,
            "pred_triage": pred_triage,
            "gold_reason": row["golden_escalation_reason"],
            "pred_reason": pred["reason"],
            "agent_reply": pred["reply"],
            "historical_reply": row["historical_brand_reply"],
            "judge_overall": j_eval["overall_score"]
        }

        # 1. Missed Escalation (Critical Safety Failure)
        if gold_triage == "ESCALATE" and pred_triage == "AUTO_HANDLE":
            failures["missed_escalations"].append(record)

        # 2. False Escalation (Cost / Queue Bloat)
        elif gold_triage == "AUTO_HANDLE" and pred_triage == "ESCALATE":
            failures["false_escalations"].append(record)

        # 3. Multi-Intent Borderline cases
        if tier == "difficult" and len(row.get("all_applicable_intents", [])) >= 2:
            failures["multi_intent_borderline"].append(record)

        # 4. Adversarial Edge Cases
        if tier == "adversarial":
            failures["adversarial_edge_cases"].append(record)

        # 5. Reply Quality / Grounding Failures
        if not j_eval["passed"]:
            record["judge_details"] = j_eval
            failures["reply_quality_failures"].append(record)

    # Save to artifacts
    out_path = ARTIFACTS_DIR / "failure_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(f"\n[Failure Analysis Complete] Logged failure categories:")
    print(f"  Missed Escalations (Safety Hazard)    : {len(failures['missed_escalations'])}")
    print(f"  False Escalations (Over-escalation)   : {len(failures['false_escalations'])}")
    print(f"  Multi-Intent Borderline Cases         : {len(failures['multi_intent_borderline'])}")
    print(f"  Adversarial Cases Analyzed            : {len(failures['adversarial_edge_cases'])}")
    print(f"  Reply Quality Failures (Judge Failed) : {len(failures['reply_quality_failures'])}")
    print(f"Saved failure report to {out_path}")

    return failures

if __name__ == "__main__":
    run_failure_analysis()
