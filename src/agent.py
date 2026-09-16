"""
Phase 5: Production-Minded AI Customer Support Agent for @AmazonHelp.
Architecture:
1. Preprocessing & Normalization
2. Calibrated Statistical Intent Classifier (TF-IDF + LogisticRegression) with Priority Hierarchy
3. Hybrid Historical Case Retrieval (55k KB) with Cosine Similarity
4. Deterministic Safety-First Triage Engine ("Maximize safe resolution, not automation rate")
5. Functional Retrieval-Conditioned Synthesis (Dynamic brand voice, carrier advice, safe portal anchors)
6. Active Response Sanitizer (Zero PII solicitation guardrail)
7. Stateful Multi-Turn Conversation Memory (Thread tracking & order lookup continuity)
"""
import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    INTENTS,
    INTENT_PRIORITY,
    DECISION_AUTO_HANDLE,
    DECISION_ESCALATE,
    ESCALATION_CATEGORIES,
    ARTIFACTS_DIR,
    RANDOM_SEED
)
from src.taxonomy import (
    INTENT_METADATA,
    HIGH_RISK_PATTERNS,
    DOMAIN_INTENT_PATTERNS,
    NEGATION_BILLING_PAT,
    NEGATION_REFUND_PAT,
    resolve_intent_collision,
    evaluate_deterministic_risk,
    detect_domain_intents,
    mask_pii
)
from src.retriever import HistoricalRetriever

logger = logging.getLogger(__name__)

AGENT_MODEL_CACHE_PATH = ARTIFACTS_DIR / "agent_intent_model.pkl"

# Custom Stop Words: preserve semantic negation tokens while pruning high-frequency preprocessor artifacts
NEGATION_STOP_WORDS = {"not", "no", "never", "neither", "nor", "none", "cannot", "without", "nothing"}
CUSTOM_STOP_WORDS = list((ENGLISH_STOP_WORDS - NEGATION_STOP_WORDS).union({"user", "link", "http", "https"}))

# Module-level pre-compiled regex patterns for zero per-request compilation latency
HUMAN_AGENT_REQUEST_PAT = re.compile(
    r"\b(human agent|talk to a (human|person|agent|representative)|"
    r"speak with (a )?(human|person|agent|representative)|customer care executive|"
    r"connect (me )?to (an? )?agent|transfer (me )?to (an? )?agent|real person|live agent|"
    r"representative)\b",
    re.I
)

DELIVERY_STOLEN_MISSING_PAT = re.compile(
    r"\b(shows?|says?|marked|claims?)\s+(as\s+)?delivered\b.*\b(not\s+(here|received|arrived)|"
    r"never\s+(left|received|got)|missing|nowhere|stole|theft|empty)\b|"
    r"\bdelivered\b.*\b(not\s+received|nowhere\s+to\s+be\s+found|didn't\s+get|never got|wrong address)\b|"
    r"\b(stole|stolen|theft|thief|porch pirate|box was empty|carrier stole|"
    r"delivered to (the )?wrong address)\b",
    re.I
)

DAMAGED_DELIVERY_ISSUE_PAT = re.compile(
    r"\b(delivered|porch|doorstep|stole|stolen|theft|missing|empty box|never arrived)\b",
    re.I
)

ORDER_CHANGE_DISPATCH_PAT = re.compile(
    r"\b(delivered to (the )?wrong address|already delivered|already shipped|too late|on the way|in transit)\b",
    re.I
)

WRONG_ADDRESS_CHECK_PAT = re.compile(r"wrong address", re.I)

REFUND_DELAY_QUERY_PAT = re.compile(
    r"\b(where is my refund|haven't received refund|still waiting for money|refund delay|"
    r"mera refund|paisa wapas|refund nahi mila)\b",
    re.I
)

CUSTOMER_HOSTILITY_PAT = re.compile(
    r"\b(terrible|worst|disgusted|furious|lawyer|police|sue|unacceptable)\b",
    re.I
)

AGENT_SIGNATURE_PAT = re.compile(r"\^([a-zA-Z]{2,3})$")
NEIGHBOR_CHECK_PAT = re.compile(r"\b(neighbors?|household|around your (property|porch))\b", re.I)
HUMAN_REP_INQUIRY_PAT = re.compile(r"\b(human|agent|person|representative)\b", re.I)

ORDER_NUMBER_PAT = re.compile(
    r"\b\d{3}-\d{7}-\d{7}\b|\b(order\s*(number|id|#)?\s*[:=]?\s*[A-Z0-9-]{8,})\b",
    re.I
)

CREDENTIAL_SOLICIT_PAT = re.compile(
    r"\b(please\s+(tweet|post|share|send|provide)\s+(us\s+)?(your\s+)?(credit card|cvv|password|full card number|card details))\b.*",
    re.I
)

AGENT_TAG_SUFFIX_PAT = re.compile(r"\^[a-zA-Z]{2,3}$")
NORMALIZE_USER_PAT = re.compile(r"@\d+")
NORMALIZE_URL_PAT = re.compile(r"https?://\S+")
NORMALIZE_SPACES_PAT = re.compile(r"\s+")

# Modular Policy Handlers for Triage Engine (Table-driven Strategy Pattern)
def _triage_account_security(customer_text: str) -> Tuple[str, str, str]:
    return (
        DECISION_ESCALATE,
        "ACCOUNT_SECURITY_RISK",
        "Suspected account compromise or security lockout requires identity verification via private channel."
    )

def _triage_billing_subscription(customer_text: str) -> Tuple[str, str, str]:
    return (
        DECISION_ESCALATE,
        "FINANCIAL_OR_BILLING_DISPUTE",
        "Recurring subscription charge or unrecognized fee requires account billing review."
    )

def _triage_damaged_wrong_missing(customer_text: str) -> Tuple[str, str, str]:
    if DAMAGED_DELIVERY_ISSUE_PAT.search(customer_text):
        return (
            DECISION_ESCALATE,
            "LOST_OR_STOLEN_DELIVERY",
            "Package reported missing, stolen, or delivered but unreceived; requires carrier investigation."
        )
    return (
        DECISION_ESCALATE,
        "DAMAGED_PHYSICAL_MERCHANDISE",
        "Customer reports damaged merchandise or incorrect item; requires replacement/refund approval."
    )

def _triage_delivery_status(customer_text: str) -> Tuple[str, str, str]:
    if DELIVERY_STOLEN_MISSING_PAT.search(customer_text):
        return (
            DECISION_ESCALATE,
            "LOST_OR_STOLEN_DELIVERY",
            "Package marked delivered but not received by customer; requires carrier check and account-specific trace."
        )
    return (
        DECISION_AUTO_HANDLE,
        "NONE",
        "General delivery status and tracking ETA guidance can be provided via self-service 'Your Orders' tracking link."
    )

def _triage_order_change(customer_text: str) -> Tuple[str, str, str]:
    if ORDER_CHANGE_DISPATCH_PAT.search(customer_text):
        cat = (
            "LOST_OR_STOLEN_DELIVERY" 
            if WRONG_ADDRESS_CHECK_PAT.search(customer_text) 
            else "MANUAL_REFUND_OR_RETURN_OVERRIDE"
        )
        return (
            DECISION_ESCALATE,
            cat,
            "Package already dispatched or delivered to wrong address; requires carrier intercept or supervisor trace."
        )
    return (
        DECISION_AUTO_HANDLE,
        "NONE",
        "Self-service order cancellation or address modification instructions can be handled via 'Your Orders' pre-dispatch."
    )

def _triage_refund_return(customer_text: str) -> Tuple[str, str, str]:
    if REFUND_DELAY_QUERY_PAT.search(customer_text):
        return (
            DECISION_ESCALATE,
            "MANUAL_REFUND_OR_RETURN_OVERRIDE",
            "Refund disbursement inquiry requiring customer account financial lookup."
        )
    return (
        DECISION_AUTO_HANDLE,
        "NONE",
        "Self-service return initiation and prepaid return label guidance can be auto-handled via 'Your Orders'."
    )

def _triage_technical_support(customer_text: str) -> Tuple[str, str, str]:
    return (
        DECISION_AUTO_HANDLE,
        "NONE",
        "First-line technical troubleshooting (rebooting device, clearing app cache) can be safely auto-handled."
    )

def _triage_feedback_complaint(customer_text: str) -> Tuple[str, str, str]:
    if CUSTOMER_HOSTILITY_PAT.search(customer_text):
        return (
            DECISION_ESCALATE,
            "CUSTOMER_AGITATION_OR_LEGAL_THREAT",
            "Severe customer agitation or brand grievance requiring human supervisor attention."
        )
    return (
        DECISION_AUTO_HANDLE,
        "NONE",
        "General customer feedback or commentary acknowledged with empathetic brand messaging."
    )

INTENT_POLICY_HANDLERS: Dict[str, Callable[[str], Tuple[str, str, str]]] = {
    "ACCOUNT_SECURITY_ACCESS": _triage_account_security,
    "BILLING_SUBSCRIPTION_PRIME": _triage_billing_subscription,
    "DAMAGED_WRONG_MISSING": _triage_damaged_wrong_missing,
    "DELIVERY_STATUS_DELAY": _triage_delivery_status,
    "ORDER_CHANGE_CANCEL": _triage_order_change,
    "REFUND_RETURN_EXCHANGE": _triage_refund_return,
    "TECHNICAL_PRODUCT_SUPPORT": _triage_technical_support,
    "FEEDBACK_COMPLAINT_GENERAL": _triage_feedback_complaint,
}

class AmazonSupportAgent:
    def __init__(
        self, 
        retriever: Optional[HistoricalRetriever] = None, 
        model_name: str = "calibrated-ml-hybrid",
        force_retrain: bool = False
    ):
        self.force_retrain = force_retrain
        self.retriever = retriever or HistoricalRetriever(force_rebuild=force_retrain)
        self.model_name = model_name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self._train_or_load_classifier()

    def _train_or_load_classifier(self) -> None:
        """Loads or trains a calibrated TF-IDF + Logistic Regression intent classifier."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.force_retrain and AGENT_MODEL_CACHE_PATH.exists():
            try:
                with open(AGENT_MODEL_CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.vectorizer = data["vectorizer"]
                    self.classifier = data["classifier"]
                return
            except Exception as exc:
                logger.warning(
                    "Agent model cache incompatible or corrupted (%s). Self-healing: retraining stratified calibrated model from corpus...",
                    exc
                )

        logger.info("Training Calibrated Stratified TF-IDF + Logistic Regression Intent Classifier...")
        class_buckets: Dict[str, List[str]] = {c: [] for c in DOMAIN_INTENT_PATTERNS}
        target_per_class = 1200
        
        # Balanced stratified sampling across 55k KB without forced defaulting
        for text in self.retriever.df_kb["customer_text"]:
            detected = detect_domain_intents(text)
            if detected:
                primary = resolve_intent_collision(detected)
                if len(class_buckets[primary]) < target_per_class:
                    clean_train = NEGATION_BILLING_PAT.sub("", text)
                    clean_train = NEGATION_REFUND_PAT.sub("", clean_train)
                    class_buckets[primary].append(self.preprocess(clean_train))

        texts: List[str] = []
        labels: List[str] = []
        for intent_name, sample_texts in class_buckets.items():
            texts.extend(sample_texts)
            labels.extend([intent_name] * len(sample_texts))

        logger.info(f"Assembled {len(texts)} balanced training instances across {len(class_buckets)} intents.")

        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words=CUSTOM_STOP_WORDS,
            sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(texts)
        
        base_clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            C=1.0
        )
        self.classifier = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=5
        )
        self.classifier.fit(X, labels)

        try:
            tmp_model_path = AGENT_MODEL_CACHE_PATH.with_suffix(".tmp")
            with open(tmp_model_path, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier
                }, f)
            import os
            os.replace(tmp_model_path, AGENT_MODEL_CACHE_PATH)
            logger.info("Calibrated model trained and cached successfully.")
        except Exception as exc:
            logger.warning("Could not cache model to %s (%s). Model remains active in memory.", AGENT_MODEL_CACHE_PATH, exc)

    def preprocess(self, text: str) -> str:
        """Normalizes user handles, urls, and redundant whitespace."""
        text = NORMALIZE_USER_PAT.sub("@User", text)
        text = NORMALIZE_URL_PAT.sub("[link]", text)
        text = NORMALIZE_SPACES_PAT.sub(" ", text).strip()
        return text

    def classify_intent(self, text: str) -> Tuple[str, float, List[str]]:
        """
        Calibrated Hybrid Intent Classification:
        1. Strips negated spans before vectorization to eliminate false keyword attraction.
        2. Computes mathematically calibrated posterior probabilities P(intent | text) via CalibratedClassifierCV.
        3. Detects domain-specific linguistic signals across all 8 intents.
        4. Blends domain signals with ML probabilities:
           - Safety priority: ACCOUNT_SECURITY_ACCESS always takes precedence when detected.
           - Collision resolution: when multiple domain intents fire, select the candidate with highest calibrated ML posterior.
           - Single domain match: verified domain candidate with posterior probability.
           - Open domain: top ML predicted intent.
        """
        detected = detect_domain_intents(text)
        
        # Strip negated spans to prevent false feature attraction in TF-IDF
        text_for_ml = NEGATION_BILLING_PAT.sub("", text)
        text_for_ml = NEGATION_REFUND_PAT.sub("", text_for_ml)
        
        # Statistical Calibrated ML prediction
        X_vec = self.vectorizer.transform([text_for_ml])
        probs = self.classifier.predict_proba(X_vec)[0]
        classes = self.classifier.classes_
        prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
        ml_top_intent = classes[probs.argmax()]
        ml_confidence = float(probs.max())

        # Hybrid Decision: combine ML probability with domain hierarchy
        if not detected:
            primary_intent = ml_top_intent
            confidence = ml_confidence
        elif "ACCOUNT_SECURITY_ACCESS" in detected:
            primary_intent = "ACCOUNT_SECURITY_ACCESS"
            confidence = prob_dict.get(primary_intent, ml_confidence)
        elif len(detected) == 1:
            primary_intent = detected[0]
            confidence = prob_dict.get(primary_intent, ml_confidence)
        else:
            primary_intent = max(detected, key=lambda c: prob_dict.get(c, 0.0))
            confidence = prob_dict.get(primary_intent, ml_confidence)

        return primary_intent, round(confidence, 4), detected

    def triage_decision(self, customer_text: str, intent: str, confidence: float) -> Tuple[str, str, str]:
        """
        Deterministic Safety Engine:
        Principle: "Maximize safe resolution, not automation rate."
        Safety guardrails CANNOT be overridden by probabilistic models.
        """
        # 0. Calibrated Confidence Gating (OOD / Low Confidence Protection)
        # Prevents low-confidence hallucinations or out-of-distribution gibberish from auto-handling
        if confidence < 0.40:
            return (
                DECISION_ESCALATE,
                "AMBIGUOUS_INQUIRY_NEEDS_CLARIFICATION",
                f"Model calibrated confidence ({confidence:.2f}) below threshold (<0.40); routed to human specialist for clarification."
            )

        # 1. Check Deterministic High-Risk Triggers
        is_risk, risk_cat, risk_reason = evaluate_deterministic_risk(customer_text)
        if is_risk:
            return DECISION_ESCALATE, risk_cat, risk_reason

        # 2. Check for Explicit Human Agent Requests (Hiver Core Support Ethos)
        if HUMAN_AGENT_REQUEST_PAT.search(customer_text):
            return (
                DECISION_ESCALATE, 
                "CUSTOMER_AGITATION_OR_LEGAL_THREAT", 
                "Customer explicitly requested human support representative intervention."
            )

        # 3. Intent-Specific Policy Rules dispatched via strategy table
        handler = INTENT_POLICY_HANDLERS.get(intent, _triage_feedback_complaint)
        return handler(customer_text)

    def generate_grounded_reply(
        self, 
        customer_text: str, 
        intent: str, 
        decision: str, 
        escalation_cat: str, 
        evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Functional Retrieval-Conditioned Synthesis:
        Extracts authentic resolution context, carrier cues, and agent sign-off tags
        from retrieved historical resolution cases (evidence[0]) and synthesizes a
        policy-compliant, privacy-safe response with verified link anchors.
        """
        top_sim = evidence[0].get("similarity", 0.0) if evidence else 0.0
        hist_reply = evidence[0]["support_reply"] if (evidence and top_sim >= 0.12) else ""
        
        # 1. Grounding: Extract authentic historical agent signature tag (e.g. ^GR, ^LL, ^CS)
        agent_tag = "^CS"
        if hist_reply:
            tag_match = AGENT_SIGNATURE_PAT.search(hist_reply.strip())
            if tag_match:
                agent_tag = f"^{tag_match.group(1)}"

        # 2. Dynamic Resolution Grounding: Extract carrier mentions & helpful advice
        carrier_hint = "the carrier"
        for c in ["USPS", "UPS", "FedEx", "Royal Mail", "Hermes"]:
            if c.lower() in hist_reply.lower():
                carrier_hint = c
                break

        neighbor_check = ""
        if NEIGHBOR_CHECK_PAT.search(hist_reply):
            neighbor_check = "We suggest checking around your property and with neighbors in the meantime. "

        if decision == DECISION_ESCALATE:
            if escalation_cat == "AMBIGUOUS_INQUIRY_NEEDS_CLARIFICATION":
                return (
                    "We want to ensure you get the exact help you need. Could you please share a few more details "
                    f"or your specific inquiry via direct message at [link] so a support specialist can assist you directly? {agent_tag}"
                )
            elif escalation_cat == "LOST_OR_STOLEN_DELIVERY":
                return (
                    "I'm so sorry to hear your package hasn't turned up even though it's marked as delivered! "
                    f"{neighbor_check}For your privacy, please do not post your order details here. Please send us a direct message "
                    f"with your order number and email address through our secure link [link] so we can investigate with {carrier_hint} right away. {agent_tag}"
                )
            elif escalation_cat == "ACCOUNT_SECURITY_RISK":
                return (
                    "Thank you for bringing this to our attention. We take account security very seriously. "
                    "Please do not share your password or payment details publicly. We strongly advise resetting your password immediately, "
                    f"and please reach out to our account security specialists directly via our secure portal: [link]. {agent_tag}"
                )
            elif escalation_cat == "FINANCIAL_OR_BILLING_DISPUTE":
                return (
                    "I understand your concern regarding unexpected charges on your statement. "
                    "Because this involves private billing and card details, please send us a DM or contact us via our secure page: [link] "
                    f"so an account specialist can safely review the transactions for you. {agent_tag}"
                )
            elif escalation_cat == "DAMAGED_PHYSICAL_MERCHANDISE":
                return (
                    "I'm truly sorry your order arrived damaged! We want to make this right immediately. "
                    f"Please send us a direct message with your order number via [link] so an account specialist can investigate and arrange a replacement or refund for you. {agent_tag}"
                )
            elif escalation_cat == "ACCOUNT_SPECIFIC_PII_REQUIRED":
                return (
                    "For your security, please delete any personal or payment details posted publicly immediately. "
                    "We take privacy very seriously and never ask for credentials on a public feed. "
                    f"Please connect with an account specialist privately through our verified portal at [link]. {agent_tag}"
                )
            else: # Customer agitation, legal threat, or explicit human agent request
                if HUMAN_REP_INQUIRY_PAT.search(customer_text):
                    return (
                        "I understand you'd like to connect directly with a representative. We are here to help! "
                        f"Please connect with us via direct message at [link] so a specialist can review your account and assist you in real time. {agent_tag}"
                    )
                return (
                    "I'm very sorry for the frustrating experience you've had. This is definitely not the standard we aim to deliver. "
                    f"Please connect with us via direct message at [link] so we can have a representative review your account history and resolve this for you. {agent_tag}"
                )

        else: # AUTO_HANDLE
            if intent == "DELIVERY_STATUS_DELAY":
                return (
                    "Sorry to hear your delivery is running behind schedule! You can track real-time courier updates and your latest delivery estimate "
                    f"directly from your account by visiting Your Orders: [link]. Let us know if we can assist further! {agent_tag}"
                )
            elif intent == "REFUND_RETURN_EXCHANGE":
                return (
                    "You can easily initiate a return or replacement directly through your account! "
                    f"Just visit 'Your Orders' at [link], select the item, and choose 'Return or replace items' to print a prepaid return label. {agent_tag}"
                )
            elif intent == "ORDER_CHANGE_CANCEL":
                return (
                    "If your order has not yet entered the shipping process, you can cancel it or update your shipping address by visiting "
                    f"'Your Orders' at [link] and clicking 'Cancel items' or 'Change'. {agent_tag}"
                )
            elif intent == "TECHNICAL_PRODUCT_SUPPORT":
                return (
                    "Sorry for the technical glitch! A quick troubleshooting step that often resolves this is restarting your device, "
                    f"checking for app updates, or clearing the app cache. Step-by-step device troubleshooting guides are available here: [link]. {agent_tag}"
                )
            else: # FEEDBACK_COMPLAINT_GENERAL
                return (
                    "Thank you for sharing your feedback with us. We appreciate hearing from our customers as it helps us improve our service. "
                    f"If there is an active order you need assistance with, please let us know! {agent_tag}"
                )

    def sanitize_reply(self, reply: str) -> str:
        """
        Final Safety & Privacy Guardrail:
        - Ensures no public solicitation of sensitive customer data.
        - Redacts actual credit card PANs, CVVs, SSNs, and passwords via mask_pii().
        - Guarantees valid sanitized link placeholders.
        - Appends brand sign-off if omitted.
        """
        cleaned = CREDENTIAL_SOLICIT_PAT.sub("please connect privately via our secure link [link]", reply)
        cleaned = mask_pii(cleaned)
        if not AGENT_TAG_SUFFIX_PAT.search(cleaned.strip()):
            cleaned = cleaned.strip() + " ^CS"
        return cleaned

    def process(
        self, 
        customer_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes inbound customer message with stateful multi-turn conversation memory.
        """
        cleaned_text = self.preprocess(customer_text)

        # Multi-turn context resolution
        is_order_number = bool(ORDER_NUMBER_PAT.search(customer_text))
        if conversation_history and len(conversation_history) > 0:
            last_turn = conversation_history[-1]
            last_agent_text = last_turn.get("agent_reply", "")
            
            # Follow-up: customer provides requested order number
            if is_order_number and ("order number" in last_agent_text.lower() or "direct message" in last_agent_text.lower()):
                prior_confidence = float(last_turn.get("intent_confidence", 0.95))
                return {
                    "session_id": session_id,
                    "intent": last_turn.get("intent", "DELIVERY_STATUS_DELAY"),
                    "intent_confidence": round(prior_confidence, 4),
                    "decision": DECISION_ESCALATE,
                    "escalation_category": last_turn.get("escalation_category", "ACCOUNT_SPECIFIC_PII_REQUIRED"),
                    "reason": "Multi-turn context: customer provided order number for active escalation inquiry.",
                    "reply": "Thank you for providing your order details. For your security, please delete any public tweets containing your order ID. An account specialist has received your information and is actively reviewing the trace. We will update you via secure channel. ^CS",
                    "evidence": [],
                    "multi_turn": True
                }

        # 1. Calibrated Intent Classification
        intent, confidence, all_detected = self.classify_intent(cleaned_text)
        
        # 2. Hybrid Historical Case Retrieval
        hits = self.retriever.retrieve(cleaned_text, top_k=2)
        evidence = [
            {
                "conversation_id": h["conversation_id"],
                "similarity": h["similarity_score"],
                "support_reply": h["support_reply"],
                "customer_text": mask_pii(h.get("customer_text", ""))
            }
            for h in hits
        ]
        
        # 3. Deterministic Safety-First Triage
        decision, category, reason = self.triage_decision(cleaned_text, intent, confidence)
        
        # 4. Functional Grounded Reply Generation
        raw_reply = self.generate_grounded_reply(cleaned_text, intent, decision, category, evidence)
        
        # 5. Final Response Sanitization
        sanitized_reply = self.sanitize_reply(raw_reply)
        
        return {
            "session_id": session_id,
            "intent": intent,
            "intent_confidence": confidence,
            "decision": decision,
            "escalation_category": category,
            "reason": reason,
            "reply": sanitized_reply,
            "evidence": evidence,
            "multi_turn": bool(conversation_history and len(conversation_history) > 0)
        }

if __name__ == "__main__":
    agent = AmazonSupportAgent()
    
    test_cases = [
        "When will my package be delivered?",
        "Package says delivered on the app, but there is nothing on my porch! I think it was stolen.",
        "Someone stole my package from my doorstep",
        "I want to talk to a human agent",
        "My package was delivered to the wrong address",
        "Mera refund kab aayega?"
    ]
    
    for tc in test_cases:
        print("\n" + "="*80)
        print("CUSTOMER:", tc)
        res = agent.process(tc)
        print("AGENT OUTPUT:")
        print(json.dumps(res, indent=2))
