"""
Intent Taxonomy Definitions, Decision Hierarchy, and Deterministic Risk Rules for @AmazonHelp.
"""
import re
from typing import Dict, List, Tuple
from src.config import (
    INTENTS,
    INTENT_PRIORITY,
    DECISION_AUTO_HANDLE,
    DECISION_ESCALATE
)

# Formal Intent Descriptions and Boundary Rules
INTENT_METADATA = {
    "DELIVERY_STATUS_DELAY": {
        "description": "Inquiries regarding transit status, tracking numbers, delayed packages, or estimated delivery dates.",
        "in_scope": "Package in transit, carrier tracking questions, ETA requests, late shipments.",
        "out_of_scope": "Package marked delivered by carrier but not received (that is DAMAGED_WRONG_MISSING/stolen).",
        "default_triage": DECISION_AUTO_HANDLE,
        "default_reason": "General tracking and ETA guidance can be addressed with self-service tracking instructions unless loss/theft is evident."
    },
    "DAMAGED_WRONG_MISSING": {
        "description": "Reports of physically damaged goods, incorrect items delivered, empty packages, or packages marked delivered but missing.",
        "in_scope": "Broken screen, leaking bottle, received incorrect product/size, missing package marked delivered (porch pirate).",
        "out_of_scope": "Buyer remorse return of undamaged product (that is REFUND_RETURN_EXCHANGE).",
        "default_triage": DECISION_ESCALATE,
        "default_reason": "Physical package defect, missing contents, or lost delivered shipments require internal carrier check and replacement/refund authorization."
    },
    "REFUND_RETURN_EXCHANGE": {
        "description": "Requests to return an item, check refund processing timelines, schedule return pickups, or exchange an item.",
        "in_scope": "Return window policy, return shipping label, refund status after dropping off return, size exchange.",
        "out_of_scope": "Disputing a recurring charge on a credit card (that is BILLING_SUBSCRIPTION_PRIME).",
        "default_triage": DECISION_AUTO_HANDLE,
        "default_reason": "Standard return window guidance and self-service return instructions can be provided automatically."
    },
    "ORDER_CHANGE_CANCEL": {
        "description": "Requests to cancel an existing order, update shipping address, or modify order parameters before dispatch.",
        "in_scope": "Cancelling order pre-dispatch, fixing shipping address typo, changing payment card on open order.",
        "out_of_scope": "Cancelling after the package has already been delivered (that is REFUND_RETURN_EXCHANGE).",
        "default_triage": DECISION_AUTO_HANDLE,
        "default_reason": "Self-service cancellation or address update links via 'Your Orders' can be auto-handled if prior to dispatch."
    },
    "BILLING_SUBSCRIPTION_PRIME": {
        "description": "Queries or disputes regarding Prime membership charges, recurring subscription fees, or unrecognized card debits.",
        "in_scope": "Charged for Prime without consent, unexpected renewal fee, digital subscription billing (Audible, Kindle Unlimited).",
        "out_of_scope": "Refund for returned merchandise (that is REFUND_RETURN_EXCHANGE).",
        "default_triage": DECISION_ESCALATE,
        "default_reason": "Financial disputes and unrecognized charges require customer identity verification and financial account review."
    },
    "ACCOUNT_SECURITY_ACCESS": {
        "description": "Reports of compromised accounts, 2FA/OTP failures, password lockouts, or suspected phishing/fraudulent emails.",
        "in_scope": "Account locked, suspicious email claiming to be Amazon, unauthorized orders, 2-step verification issues.",
        "out_of_scope": "General password change request via self-service reset portal.",
        "default_triage": DECISION_ESCALATE,
        "default_reason": "Account compromise and security risks require immediate human escalation and private verification via secure channel."
    },
    "TECHNICAL_PRODUCT_SUPPORT": {
        "description": "Troubleshooting hardware devices (Kindle, Echo, Fire TV) or digital services (Amazon app glitches, promo code bugs).",
        "in_scope": "Kindle screen frozen, Fire stick not connecting to WiFi, Amazon app crash, promo code not applying at checkout.",
        "out_of_scope": "Returning a broken Kindle for a cash refund (that is DAMAGED_WRONG_MISSING or REFUND_RETURN_EXCHANGE).",
        "default_triage": DECISION_AUTO_HANDLE,
        "default_reason": "First-line troubleshooting steps (reboot, clear app cache) can be auto-handled safely."
    },
    "FEEDBACK_COMPLAINT_GENERAL": {
        "description": "General praise, policy rants, broad brand commentary, or inquiries lacking order-specific context.",
        "in_scope": "Expressing frustration at customer service in general, sarcasm, pricing complaints without order details.",
        "out_of_scope": "Specific problem statement with an identifiable order/delivery issue.",
        "default_triage": DECISION_AUTO_HANDLE,
        "default_reason": "General feedback and broad inquiries can be auto-handled with standard empathetic acknowledgment unless hostile escalation is required."
    }
}

# Deterministic Risk Patterns for Triage Safety Engine
# Critical rule: LLM cannot override deterministic safety rules!
HIGH_RISK_PATTERNS = {
    "ACCOUNT_SECURITY_RISK": re.compile(
        r"\b(hacked|fraud|compromised|phishing|unauthorized access|scam|stole my identity|otp bypass|someone accessed)\b", 
        re.I
    ),
    "LOST_OR_STOLEN_DELIVERY": re.compile(
        r"\b(says delivered|marked delivered|never arrived|stolen|porch pirate|box was empty|carrier stole|not received but delivered)\b", 
        re.I
    ),
    "DAMAGED_PHYSICAL_MERCHANDISE": re.compile(
        r"\b(shattered|smashed|completely broken|damaged goods|ruined|defective screen|leaking everywhere)\b", 
        re.I
    ),
    "FINANCIAL_OR_BILLING_DISPUTE": re.compile(
        r"\b(unauthorized charge|charged my card|double charged|unrecognized charge|bank fee|stolen credit card|chargeback)\b", 
        re.I
    ),
    "CUSTOMER_AGITATION_OR_LEGAL_THREAT": re.compile(
        r"\b(lawyer|attorney|police|court|legal action|consumer court|better business bureau|bbb complaint|lawsuit|unacceptable service|furious|disgusted)\b", 
        re.I
    )
}

def resolve_intent_collision(detected_intents: List[str]) -> str:
    """
    Applies the explicit priority hierarchy:
    Security > Damaged/Missing > Billing > Refund/Return > Order Change > Delivery > Technical > General
    """
    if not detected_intents:
        return "FEEDBACK_COMPLAINT_GENERAL"
    for priority_intent in INTENT_PRIORITY:
        if priority_intent in detected_intents:
            return priority_intent
    return detected_intents[0]

def evaluate_deterministic_risk(customer_text: str) -> Tuple[bool, str, str]:
    """
    Scans text for deterministic high-risk triggers.
    Returns: (is_high_risk, escalation_category, reason)
    """
    for category, pattern in HIGH_RISK_PATTERNS.items():
        if pattern.search(customer_text):
            reason = f"Deterministic safety rule triggered: {category.lower().replace('_', ' ')} detected in customer message."
            return True, category, reason
    return False, "NONE", ""
