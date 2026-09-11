"""
Phase 7: Deterministic Multi-Criteria Reply Evaluation Rubric.
Evaluates:
1. Groundedness & Domain Alignment (1-5)
2. Brand Tone & Empathy (1-5)
3. Actionability & Guidance (1-5)
4. Channel & PII Safety (1-5)
5. Overall Pass / Fail (Binary: Groundedness >= 4, Actionability >= 4, PII Safety >= 4, Overall Score >= 3.8)

Design: Explicit, transparent rule-based scoring engine for reproducible offline evaluation.
"""
import re
import pandas as pd
from typing import Dict, Any, List, Tuple


class ReplyQualityJudge:
    def __init__(self):
        pass

    def evaluate_reply(self, customer_text: str, reply: str, intent: str, decision: str) -> Dict[str, Any]:
        """
        Calibrated Multi-Criteria Quality Evaluation Rubric:
        - Groundedness (1-5)
        - Brand Tone & Empathy (1-5)
        - Actionability (1-5)
        - PII & Channel Safety (1-5)
        - Overall Pass (bool)
        """
        c_lower = customer_text.lower()
        r_lower = reply.lower()

        # 1. PII & Channel Safety (Strict)
        if re.search(r"\b(credit card|cvv|password|full card number)\b", r_lower):
            pii_safety = 1
        elif decision == "ESCALATE" and not re.search(r"\b(dm|direct message|private|link|contact us)\b", r_lower):
            pii_safety = 2
        else:
            pii_safety = 5

        # 2. Brand Tone & Empathy
        empathy_markers = [
            "sorry", "apologize", "apologies", "understand", "hate to hear", 
            "frustrating", "glad to help", "happy to help", "make this right"
        ]
        has_empathy = any(m in r_lower for m in empathy_markers)
        has_signoff = bool(re.search(r"\^[a-zA-Z]{2,3}$", reply.strip()))
        
        if has_empathy and has_signoff:
            tone = 5
        elif has_empathy or has_signoff:
            tone = 4
        elif len(reply.split()) > 10:
            tone = 3
        else:
            tone = 2

        # 3. Actionability
        action_markers = ["visit", "click", "dm", "send us", "select", "restart", "track", "step", "portal", "link"]
        action_count = sum(1 for m in action_markers if m in r_lower)
        if action_count >= 2:
            actionability = 5
        elif action_count == 1:
            actionability = 4
        else:
            actionability = 2

        # 4. Groundedness
        intent_grounding = {
            "DELIVERY_STATUS_DELAY": ["track", "order", "delivery", "carrier", "arrive", "link", "schedule"],
            "DAMAGED_WRONG_MISSING": ["damaged", "broken", "package", "replacement", "refund", "investigate", "dm", "order"],
            "REFUND_RETURN_EXCHANGE": ["return", "refund", "replace", "order", "items", "label", "orders"],
            "ORDER_CHANGE_CANCEL": ["cancel", "order", "address", "shipping", "process", "orders"],
            "BILLING_SUBSCRIPTION_PRIME": ["charge", "billing", "prime", "statement", "specialist", "account", "dm"],
            "ACCOUNT_SECURITY_ACCESS": ["security", "password", "reset", "specialist", "portal", "protect"],
            "TECHNICAL_PRODUCT_SUPPORT": ["restart", "cache", "device", "button", "app", "troubleshooting"],
            "FEEDBACK_COMPLAINT_GENERAL": ["feedback", "experience", "improve", "apologize", "assist"]
        }
        
        expected_keywords = intent_grounding.get(intent, ["order", "help", "support"])
        matches = sum(1 for kw in expected_keywords if kw in r_lower)
        if matches >= 3:
            groundedness = 5
        elif matches == 2:
            groundedness = 4
        elif matches == 1:
            groundedness = 3
        else:
            groundedness = 2

        overall_score = round(0.35 * groundedness + 0.25 * actionability + 0.20 * tone + 0.20 * pii_safety, 2)
        passed = bool(groundedness >= 4 and actionability >= 4 and pii_safety >= 4 and overall_score >= 3.8)

        return {
            "groundedness": groundedness,
            "tone": tone,
            "actionability": actionability,
            "pii_safety": pii_safety,
            "overall_score": overall_score,
            "passed": passed
        }

# Backwards compatibility and descriptive aliases
ReplyQualityRubric = ReplyQualityJudge
DeterministicReplyRubric = ReplyQualityJudge

def evaluate_reply_batch(
    rubric: ReplyQualityJudge, 
    predictions: List[Dict[str, Any]], 
    golden_df: pd.DataFrame
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates a batch of agent predictions against golden reference records using the 4-criteria rubric.
    Returns (pass_rate, detailed_results).
    """
    results = [
        rubric.evaluate_reply(row["customer_text"], p["reply"], p["intent"], p["decision"])
        for p, (_, row) in zip(predictions, golden_df.iterrows())
    ]
    pass_rate = sum(1 for r in results if r["passed"]) / len(results) if results else 0.0
    return round(float(pass_rate), 4), results

if __name__ == "__main__":
    from src.config import GOLDEN_DATA_DIR
    from src.agent import AmazonSupportAgent
    golden_df = pd.read_json(GOLDEN_DATA_DIR / "golden_eval_set.json")
    agent = AmazonSupportAgent()
    print("Generating predictions for reply quality rubric check...")
    preds = [agent.process(row["customer_text"]) for _, row in golden_df.iterrows()]
    rubric = ReplyQualityRubric()
    pass_rate, results = evaluate_reply_batch(rubric, preds, golden_df)
    print(f"Reply Quality Pass Rate across {len(golden_df)} cases: {pass_rate * 100:.1f}%")
