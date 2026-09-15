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
        r"\b(hacked|fraud|compromised|phishing|unauthorized access|scam|"
        r"stole my identity|otp bypass|someone accessed|hacked ho gaya|account hack)\b",
        re.I
    ),
    "LOST_OR_STOLEN_DELIVERY": re.compile(
        r"\b(says delivered|marked delivered|never arrived|stole|stolen|steal|stealing|theft|thief|"
        r"thieves|porch pirate|box was empty|carrier stole|not received but delivered|"
        r"delivered to (the )?wrong address|delivered to (my )?neighbor|"
        r"taken from (my )?(doorstep|porch|lobby))\b|"
        r"\b(shows?|says?|marked|claims?)\s+(as\s+)?delivered\b.*\b(not\s+(here|received|arrived)|"
        r"never\s+(left|received|got)|missing|nowhere|stole|theft|empty)\b|"
        r"\bdelivered\b.*\b(not\s+received|nowhere to be found|didn't get|never got)\b",
        re.I
    ),
    "DAMAGED_PHYSICAL_MERCHANDISE": re.compile(
        r"\b(shattered|smashed|completely broken|damaged goods|ruined|defective screen|leaking everywhere)\b",
        re.I
    ),
    "FINANCIAL_OR_BILLING_DISPUTE": re.compile(
        r"\b(unauthorized charge|charged my card|double charged|unrecognized charge|bank fee|"
        r"stolen credit card|chargeback|paisa kat|paise kat|amount deduct|extra charge|unauthorized debit)\b",
        re.I
    ),
    "CUSTOMER_AGITATION_OR_LEGAL_THREAT": re.compile(
        r"\b(human agent|talk to a (human|person|agent|representative)|"
        r"speak with (a )?(human|person|agent|representative)|customer care executive|"
        r"connect (me )?to (an? )?agent|transfer (me )?to (an? )?agent|real person|live agent|"
        r"representative|lawyer|attorney|police|court|legal action|consumer court|"
        r"better business bureau|bbb complaint|lawsuit|unacceptable service|furious|disgusted)\b",
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

# High-precision linguistic domain patterns for intent feature extraction & collision handling
DOMAIN_INTENT_PATTERNS = {
    "ACCOUNT_SECURITY_ACCESS": re.compile(
        r"\b(hacked|fraud|unauthorized|phishing|otp|locked out|password|scam|security|"
        r"compromised|login|someone else accessed|hacked ho gaya|account hack)\b",
        re.I
    ),
    "DAMAGED_WRONG_MISSING": re.compile(
        r"\b(damaged|broken|empty box|missing|wrong item|defective|shattered|cracked|ruined|"
        r"torn|opened package|stole|stolen|steal|stealing|theft|thief|never arrived|"
        r"delivered to (the )?wrong address|porch pirate)\b|"
        r"\b(shows?|says?|marked|claims?)\s+(as\s+)?delivered\b.*\b(not\s+(here|received|arrived)|"
        r"never\s+(left|received|got)|missing|stolen|theft|empty|porch|nowhere)\b|"
        r"\bdelivered\b.*\b(not\s+received|nowhere\s+to\s+be\s+found|didn't\s+get|never got|stolen|porch)\b",
        re.I
    ),
    "BILLING_SUBSCRIPTION_PRIME": re.compile(
        r"\b(prime|membership|subscription|charged|billing|debit|card charged|renew|"
        r"unrecognized charge|audible fee|annual charge|cashback|amazon pay balance|"
        r"wallet balance|promotional credit|paisa kat|paise kat)\b",
        re.I
    ),
    "REFUND_RETURN_EXCHANGE": re.compile(
        r"\b(refund|return|returning|exchange|money back|pickup|drop off|refunded|"
        r"send back|return label|replacement|mera refund|paisa wapas|refund nahi mila)\b",
        re.I
    ),
    "ORDER_CHANGE_CANCEL": re.compile(
        r"\b(cancel|cancelled|cancellation|change address|modify order|wrong address|"
        r"change payment|stop delivery)\b",
        re.I
    ),
    "DELIVERY_STATUS_DELAY": re.compile(
        r"\b(delivery|deliver|late|delayed|delay|where is|tracking|track|carrier|package|"
        r"parcel|courier|transit|arrive|arriving|eta|status|not delivered|kab aayega)\b",
        re.I
    ),
    "TECHNICAL_PRODUCT_SUPPORT": re.compile(
        r"\b(kindle|fire stick|echo|alexa|app|website|code|voucher|coupon|promo|"
        r"error|crash|bug|tv app|frozen)\b",
        re.I
    )
}

def detect_domain_intents(text: str) -> List[str]:
    """Detect all candidate intents matching domain-specific linguistic patterns."""
    detected = []
    for intent, pat in DOMAIN_INTENT_PATTERNS.items():
        if pat.search(text):
            detected.append(intent)
    return detected

