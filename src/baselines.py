"""
Phase 4: Baselines for @AmazonHelp Evaluation.
- Baseline 0 (Trivial): Majority-class intent + Constant Escalate / Auto rule + Static canned reply.
- Baseline 1 (Simple): TF-IDF + Logistic Regression Intent Classifier + Keyword Rule-Based Triage + 1-NN Raw Retrieved Historical Reply.
"""
import re
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import INTENTS, ARTIFACTS_DIR, RANDOM_SEED
from src.retriever import HistoricalRetriever

MODEL_CACHE_PATH = ARTIFACTS_DIR / "simple_baseline_model.pkl"

class TrivialBaseline:
    """
    Trivial Baseline (Baseline 0):
    - Intent: Always predicts the most common class.
    - Triage: Always predicts AUTO_HANDLE (or constant rule).
    - Reply: Static boilerplate canned reply.
    """
    def __init__(self, majority_intent: str = "DELIVERY_STATUS_DELAY"):
        self.majority_intent = majority_intent

    def predict(self, customer_text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "decision": "AUTO_HANDLE",
            "escalation_category": "NONE",
            "reason": "Trivial baseline default auto-handle decision.",
            "reply": "Hello! Thanks for reaching out to Amazon Help. Please visit amazon.com/your-orders to view the status of your order or follow self-service instructions. Let us know if you need anything else! ^CS",
            "evidence": []
        }

class SimpleBaseline:
    """
    Simple Baseline (Baseline 1):
    - Intent: TF-IDF + Logistic Regression classifier.
    - Triage: Regex-based keyword heuristic.
    - Reply: Top-1 raw historical brand reply retrieved via 1-NN search.
    """
    def __init__(self, retriever: HistoricalRetriever = None):
        self.retriever = retriever or HistoricalRetriever()
        self.vectorizer = None
        self.classifier = None
        self._train_or_load_classifier()

    def _train_or_load_classifier(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_CACHE_PATH.exists():
            print(f"[SimpleBaseline] Loading cached classifier from {MODEL_CACHE_PATH}...")
            with open(MODEL_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.classifier = data["classifier"]
            return

        print("[SimpleBaseline] Training TF-IDF + Logistic Regression Intent Classifier...")
        # To train a clean classifier, we use the taxonomy regexes on 8,000 KB samples to create pseudo-labels
        from src.build_golden_set import classify_candidate
        
        sample_kb = self.retriever.df_kb.sample(min(10000, len(self.retriever.df_kb)), random_state=RANDOM_SEED)
        texts = []
        labels = []
        
        for text in sample_kb["customer_text"]:
            prim_intent, _ = classify_candidate(text)
            texts.append(text)
            labels.append(prim_intent)
            
        self.vectorizer = TfidfVectorizer(max_features=25000, stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
        self.classifier.fit(X, labels)
        
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier
            }, f)
        print("[SimpleBaseline] Classifier trained and cached.")

    def predict(self, customer_text: str) -> Dict[str, Any]:
        # 1. Intent Classification
        X_vec = self.vectorizer.transform([customer_text])
        predicted_intent = self.classifier.predict(X_vec)[0]
        probs = self.classifier.predict_proba(X_vec)[0]
        confidence = float(max(probs))

        # 2. Simple Rule-Based Triage
        escalate_keywords = re.compile(
            r"\b(stolen|delivered|refund|cancel|hacked|fraud|charge|broken|damaged|lawyer|police|sue|unauthorized)\b", 
            re.I
        )
        if escalate_keywords.search(customer_text):
            decision = "ESCALATE"
            category = "ACCOUNT_SPECIFIC_PII_REQUIRED"
            reason = "Simple baseline keyword match detected escalation trigger."
        else:
            decision = "AUTO_HANDLE"
            category = "NONE"
            reason = "No simple escalation keywords detected."

        # 3. 1-NN Raw Historical Reply
        hits = self.retriever.retrieve(customer_text, top_k=1)
        if hits:
            reply = hits[0]["support_reply"]
            evidence = [{"conversation_id": hits[0]["conversation_id"], "similarity": hits[0]["similarity_score"]}]
        else:
            reply = "Hello! Please check your account or contact us for help. ^CS"
            evidence = []

        return {
            "intent": predicted_intent,
            "intent_confidence": round(confidence, 4),
            "decision": decision,
            "escalation_category": category,
            "reason": reason,
            "reply": reply,
            "evidence": evidence
        }

if __name__ == "__main__":
    t_base = TrivialBaseline()
    s_base = SimpleBaseline()
    
    test_q = "My package says delivered yesterday but it was never left on my porch. I want a refund."
    print("\n--- Test Query ---")
    print(test_q)
    print("\nTrivial Baseline Output:")
    print(t_base.predict(test_q))
    print("\nSimple Baseline Output:")
    print(s_base.predict(test_q))
