"""
Phase 7: Deterministic Multi-Criteria Reply Evaluation Rubric.

Evaluates 4 transparent dimensions:
1. Groundedness & Domain Alignment (1-5)
2. Brand Tone & Empathy (1-5)
3. Actionability & Guidance (1-5)
4. Channel & PII Safety (1-5)
5. Overall Pass / Fail (Binary: Groundedness >= 4, Actionability >= 4, PII Safety >= 4, Overall Score >= 3.8)

Design: Explicit, transparent rule-based scoring engine for reproducible offline evaluation.
"""
import re
from typing import Dict, Any, List, Tuple
import pandas as pd

# Module-level pre-compiled regexes for maximum evaluation throughput
PII_SENSITIVE_PAT = re.compile(r"\b(credit card|cvv|password|full card number)\b", re.I)
SECURE_CHANNEL_PAT = re.compile(r"\b(dm|direct message|private|link|contact us)\b", re.I)
AGENT_SIGNOFF_PAT = re.compile(r"\^[a-zA-Z]{2,3}$")

EMPATHY_MARKERS = [
    "sorry", "apologize", "apologies", "understand", "hate to hear",
    "frustrating", "glad to help", "happy to help", "make this right"
]

ACTION_MARKERS = [
    "visit", "click", "dm", "send us", "select", "restart", "track", "step", "portal", "link"
]

INTENT_GROUNDING_KEYWORDS: Dict[str, List[str]] = {
    "DELIVERY_STATUS_DELAY": ["track", "order", "delivery", "carrier", "arrive", "link", "schedule"],
    "DAMAGED_WRONG_MISSING": ["damaged", "broken", "package", "replacement", "refund", "investigate", "dm", "order"],
    "REFUND_RETURN_EXCHANGE": ["return", "refund", "replace", "order", "items", "label", "orders"],
    "ORDER_CHANGE_CANCEL": ["cancel", "order", "address", "shipping", "process", "orders"],
    "BILLING_SUBSCRIPTION_PRIME": ["charge", "billing", "prime", "statement", "specialist", "account", "dm"],
    "ACCOUNT_SECURITY_ACCESS": ["security", "password", "reset", "specialist", "portal", "protect"],
    "TECHNICAL_PRODUCT_SUPPORT": ["restart", "cache", "device", "button", "app", "troubleshooting"],
    "FEEDBACK_COMPLAINT_GENERAL": ["feedback", "experience", "improve", "apologize", "assist"]
}


class DeterministicQualityRubric:
    """
    Deterministic rule-based reply quality scoring engine.
    Stateless evaluator implementing the 4-criteria quality rubric.
    """

    def evaluate_reply(
        self,
        customer_text: str,
        reply: str,
        intent: str,
        decision: str
    ) -> Dict[str, Any]:
        """
        Evaluates a support reply against the 4-criteria quality rubric.

        Returns a dictionary containing individual criterion scores (1-5),
        the weighted overall score, and the binary pass/fail decision.
        """
        c_lower = customer_text.lower()
        r_lower = reply.lower()

        # 1. PII & Channel Safety (Strict)
        if PII_SENSITIVE_PAT.search(r_lower):
            pii_safety = 1
        elif decision == "ESCALATE" and not SECURE_CHANNEL_PAT.search(r_lower):
            pii_safety = 2
        else:
            pii_safety = 5

        # 2. Brand Tone & Empathy
        has_empathy = any(m in r_lower for m in EMPATHY_MARKERS)
        has_signoff = bool(AGENT_SIGNOFF_PAT.search(reply.strip()))

        if has_empathy and has_signoff:
            tone = 5
        elif has_empathy or has_signoff:
            tone = 4
        elif len(reply.split()) > 10:
            tone = 3
        else:
            tone = 2

        # 3. Actionability
        action_count = sum(1 for m in ACTION_MARKERS if m in r_lower)
        if action_count >= 2:
            actionability = 5
        elif action_count == 1:
            actionability = 4
        else:
            actionability = 2

        # 4. Groundedness
        expected_keywords = INTENT_GROUNDING_KEYWORDS.get(intent, ["order", "help", "support"])
        matches = sum(1 for kw in expected_keywords if kw in r_lower)
        if matches >= 3:
            groundedness = 5
        elif matches == 2:
            groundedness = 4
        elif matches == 1:
            groundedness = 3
        else:
            groundedness = 2

        overall_score = round(
            0.35 * groundedness + 0.25 * actionability + 0.20 * tone + 0.20 * pii_safety, 
            2
        )
        passed = bool(
            groundedness >= 4 and actionability >= 4 and pii_safety >= 4 and overall_score >= 3.8
        )

        return {
            "groundedness": groundedness,
            "tone": tone,
            "actionability": actionability,
            "pii_safety": pii_safety,
            "overall_score": overall_score,
            "passed": passed
        }


# Backwards-compatible aliases
ReplyQualityJudge = DeterministicQualityRubric
ReplyQualityRubric = DeterministicQualityRubric
DeterministicReplyRubric = DeterministicQualityRubric


def evaluate_reply_batch(
    rubric: DeterministicQualityRubric,
    predictions: List[Dict[str, Any]],
    golden_df: pd.DataFrame
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates a batch of agent predictions against golden reference records.

    Returns (pass_rate, detailed_results).
    """
    results = [
        rubric.evaluate_reply(row["customer_text"], p["reply"], p["intent"], p["decision"])
        for p, (_, row) in zip(predictions, golden_df.iterrows())
    ]
    pass_rate = sum(1 for r in results if r["passed"]) / len(results) if results else 0.0
    return round(float(pass_rate), 4), results
