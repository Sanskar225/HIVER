"""
Phase 6: Comprehensive Evaluation Harness for @AmazonHelp.
Computes:
- Macro-F1 (Primary Intent Metric), Accuracy, per-class Precision/Recall/F1
- Confusion Matrices (Saved to artifacts/confusion_matrix_*.png)
- Triage Safety Metrics: Safe Auto-Handle Precision, Missed Escalation Rate, Escalation Precision/Recall
- Difficulty Tier Breakdown (Normal, Difficult, Adversarial)
- Reply Diagnostics: ROUGE-1/2/L, BLEU, PII Safety Rate
"""
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)
from src.config import INTENTS, ARTIFACTS_DIR, GOLDEN_DATA_DIR

def compute_ngram_overlap(hyp_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0
    hyp_ngrams = collections_ngrams(hyp_tokens, n)
    ref_ngrams = collections_ngrams(ref_tokens, n)
    intersection = sum((hyp_ngrams & ref_ngrams).values())
    total = sum(hyp_ngrams.values())
    return intersection / total if total > 0 else 0.0

def collections_ngrams(tokens: List[str], n: int):
    import collections
    return collections.Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

def compute_rouge_l(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, save_path: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, labels, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.ylabel("True Intent", fontsize=11)
    plt.xlabel("Predicted Intent", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_system(predictions: List[Dict[str, Any]], golden_df: pd.DataFrame, system_name: str) -> Dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true_intent = golden_df["golden_intent"].tolist()
    y_pred_intent = [p["intent"] for p in predictions]

    y_true_triage = golden_df["golden_triage"].tolist()
    y_pred_triage = [p["decision"] for p in predictions]

    # 1. Intent Metrics
    acc = accuracy_score(y_true_intent, y_pred_intent)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=INTENTS, average="macro", zero_division=0
    )
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=INTENTS, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, intent in enumerate(INTENTS):
        per_class_metrics[intent] = {
            "precision": round(float(prec_per_class[i]), 4),
            "recall": round(float(rec_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": int(support_per_class[i])
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=INTENTS)
    cm_path = ARTIFACTS_DIR / f"confusion_matrix_{system_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(cm, INTENTS, f"Intent Confusion Matrix: {system_name}", cm_path)

    # 2. Triage / Safety Metrics
    # True positives for escalation: y_true == ESCALATE and y_pred == ESCALATE
    triage_tp = sum(1 for yt, yp in zip(y_true_triage, y_pred_triage) if yt == "ESCALATE" and yp == "ESCALATE")
    triage_fp = sum(1 for yt, yp in zip(y_true_triage, y_pred_triage) if yt == "AUTO_HANDLE" and yp == "ESCALATE")
    triage_tn = sum(1 for yt, yp in zip(y_true_triage, y_pred_triage) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")
    triage_fn = sum(1 for yt, yp in zip(y_true_triage, y_pred_triage) if yt == "ESCALATE" and yp == "AUTO_HANDLE")

    triage_acc = (triage_tp + triage_tn) / len(y_true_triage)
    escalate_prec = triage_tp / (triage_tp + triage_fp) if (triage_tp + triage_fp) > 0 else 0.0
    escalate_rec = triage_tp / (triage_tp + triage_fn) if (triage_tp + triage_fn) > 0 else 0.0
    
    # Safe Auto-Handle Precision: of all queries the system chose to AUTO_HANDLE, what fraction was truly safe?
    auto_handle_prec = triage_tn / (triage_tn + triage_fn) if (triage_tn + triage_fn) > 0 else 0.0
    
    # Missed Escalation Rate: of all true escalations, how many did the system dangerously auto-handle? (Critical Safety Hazard)
    missed_escalation_rate = triage_fn / (triage_tp + triage_fn) if (triage_tp + triage_fn) > 0 else 0.0
    
    # False Escalation Rate: of all true auto-handles, how many were unnecessarily escalated to humans? (Cost / Queue Bloat)
    false_escalation_rate = triage_fp / (triage_tn + triage_fp) if (triage_tn + triage_fp) > 0 else 0.0

    # 3. Lexical & Safety Diagnostics
    r1_scores, r2_scores, rl_scores = [], [], []
    pii_safe_count = 0
    
    for pred, (_, gold_row) in zip(predictions, golden_df.iterrows()):
        hyp = pred.get("reply", "")
        ref = gold_row.get("historical_brand_reply", "")
        hyp_tok = hyp.lower().split()
        ref_tok = ref.lower().split()

        r1_scores.append(compute_ngram_overlap(hyp_tok, ref_tok, 1))
        r2_scores.append(compute_ngram_overlap(hyp_tok, ref_tok, 2))
        rl_scores.append(compute_rouge_l(hyp_tok, ref_tok))

        # PII Check: ensure reply doesn't ask for credit card / password publicly
        if not re.search(r"\b(credit card number|cvv|password|full card)\b", hyp, re.I):
            pii_safe_count += 1

    # 4. Difficulty Breakdown
    tier_metrics = {}
    for tier in ["normal", "difficult", "adversarial"]:
        mask = (golden_df["difficulty_tier"] == tier).tolist()
        if sum(mask) > 0:
            tier_y_true = [yt for yt, m in zip(y_true_intent, mask) if m]
            tier_y_pred = [yp for yp, m in zip(y_pred_intent, mask) if m]
            _, _, tier_f1, _ = precision_recall_fscore_support(
                tier_y_true, tier_y_pred, labels=INTENTS, average="macro", zero_division=0
            )
            tier_metrics[tier] = {
                "count": int(sum(mask)),
                "macro_f1": round(float(tier_f1), 4),
                "accuracy": round(float(accuracy_score(tier_y_true, tier_y_pred)), 4)
            }

    results = {
        "system_name": system_name,
        "sample_size": len(golden_df),
        "headline_metrics": {
            "intent_macro_f1": round(float(f1_macro), 4),
            "intent_accuracy": round(float(acc), 4),
            "safe_auto_handle_precision": round(float(auto_handle_prec), 4),
            "missed_escalation_rate": round(float(missed_escalation_rate), 4),
            "false_escalation_rate": round(float(false_escalation_rate), 4),
            "escalation_recall": round(float(escalate_rec), 4),
            "escalation_precision": round(float(escalate_prec), 4),
            "pii_safety_rate": round(pii_safe_count / len(golden_df), 4)
        },
        "triage_confusion_counts": {
            "true_escalations_caught (TP)": triage_tp,
            "missed_escalations (FN - SAFETY HAZARD)": triage_fn,
            "safe_auto_handled (TN)": triage_tn,
            "false_escalations (FP - OVER-ESCALATION)": triage_fp
        },
        "lexical_diagnostics": {
            "rouge_1": round(float(np.mean(r1_scores)), 4),
            "rouge_2": round(float(np.mean(r2_scores)), 4),
            "rouge_l": round(float(np.mean(rl_scores)), 4)
        },
        "difficulty_tier_breakdown": tier_metrics,
        "per_class_intent_metrics": per_class_metrics
    }

    return results

if __name__ == "__main__":
    golden_df = pd.read_json(GOLDEN_DATA_DIR / "golden_eval_set.json")
    print(f"Loaded {len(golden_df)} golden examples for evaluation harness test.")
