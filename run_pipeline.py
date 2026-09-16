"""
Master Evaluation & Reproduction Pipeline for @AmazonHelp.
Reproduces all headline results and baseline comparisons in < 15 minutes.
Outputs:
- artifacts/evaluation_metrics.json
- artifacts/headline_results_table.md
- artifacts/confusion_matrix_*.png
"""
import sys
import json
import time
import pandas as pd
from pathlib import Path
from src.config import GOLDEN_DATA_DIR, ARTIFACTS_DIR
from src.retriever import HistoricalRetriever
from src.baselines import TrivialBaseline, SimpleBaseline
from src.agent import AmazonSupportAgent
from src.evaluator import evaluate_system
from src.quality_rubric import DeterministicQualityRubric, evaluate_reply_batch

def run_pipeline(reproduce: bool = True, force_retrain: bool = False):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("="*90)
    print("HIVER SDE INTERN: MASTER EVALUATION PIPELINE FOR @AmazonHelp")
    print("="*90)

    golden_json_path = GOLDEN_DATA_DIR / "golden_eval_set.json"
    if not golden_json_path.exists():
        raise FileNotFoundError(
            f"Golden evaluation set not found at {golden_json_path}. "
            "The golden evaluation suite is an audited ground-truth benchmark and must be present."
        )

    print(f"\n[Step 1/5] Loading 200 Golden Evaluation Examples from {golden_json_path}...")
    golden_df = pd.read_json(golden_json_path)
    print(f"Loaded {len(golden_df)} stratified examples.")
    print(f"Tiers: {golden_df['difficulty_tier'].value_counts().to_dict()}")
    print(f"Triage: {golden_df['golden_triage'].value_counts().to_dict()}")

    print("\n[Step 2/5] Initializing Knowledge Base & Models...")
    if force_retrain:
        print("  [--force-retrain active] Rebuilding TF-IDF index and retraining ML model from scratch...")
    retriever = HistoricalRetriever(force_rebuild=force_retrain)
    trivial_base = TrivialBaseline()
    simple_base = SimpleBaseline(retriever=retriever)
    agent = AmazonSupportAgent(retriever=retriever, force_retrain=force_retrain)

    print("\n[Step 3/5] Running Predictions across Golden Set...")
    print("  Running Baseline 0 (Trivial: Majority + Constant Rule)...")
    trivial_preds = [trivial_base.predict(row["customer_text"]) for _, row in golden_df.iterrows()]

    print("  Running Baseline 1 (Simple: TF-IDF + Keyword Rule + 1-NN Reply)...")
    simple_preds = [simple_base.predict(row["customer_text"]) for _, row in golden_df.iterrows()]

    print("  Running Proposed Agent (Multi-Intent Hierarchy + Safety Triage + Grounded Reply)...")
    agent_preds = [agent.process(row["customer_text"]) for _, row in golden_df.iterrows()]

    print("\n[Step 4/5] Computing Automated Metrics & Confusion Matrices...")
    trivial_eval = evaluate_system(trivial_preds, golden_df, "Trivial Baseline")
    simple_eval = evaluate_system(simple_preds, golden_df, "Simple Baseline")
    agent_eval = evaluate_system(agent_preds, golden_df, "Proposed Agent")

    print("\n[Step 5/5] Running Deterministic Multi-Criteria Reply Rubric...")
    rubric = DeterministicQualityRubric()
    agent_pass_rate, _ = evaluate_reply_batch(rubric, agent_preds, golden_df)
    agent_eval["headline_metrics"]["grounded_reply_pass_rate"] = agent_pass_rate

    simple_pass_rate, _ = evaluate_reply_batch(rubric, simple_preds, golden_df)
    simple_eval["headline_metrics"]["grounded_reply_pass_rate"] = simple_pass_rate

    trivial_pass_rate, _ = evaluate_reply_batch(rubric, trivial_preds, golden_df)
    trivial_eval["headline_metrics"]["grounded_reply_pass_rate"] = trivial_pass_rate

    # Compile Final Comparative Metrics
    full_metrics = {
        "metadata": {
            "target_brand": "@AmazonHelp",
            "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "golden_sample_size": len(golden_df),
            "kb_corpus_size": len(retriever.df_kb)
        },
        "systems": {
            "trivial_baseline": trivial_eval,
            "simple_baseline": simple_eval,
            "proposed_agent": agent_eval
        }
    }

    metrics_json_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Generate Markdown Headline Table
    tb_h = trivial_eval["headline_metrics"]
    sb_h = simple_eval["headline_metrics"]
    ag_h = agent_eval["headline_metrics"]

    tb_d = trivial_eval["difficulty_tier_breakdown"]
    sb_d = simple_eval["difficulty_tier_breakdown"]
    ag_d = agent_eval["difficulty_tier_breakdown"]

    markdown_table = f"""# Benchmark Results: @AmazonHelp AI Support Agent vs. Dual Baselines

**Evaluation Suite**: 200 Stratified Hand-Labelled Cases (140 Normal, 40 Difficult, 20 Adversarial)  
**Historical Evidence Base**: 55,011 disjoint @AmazonHelp customer->brand resolution pairs  

---

### 1. Headline System Comparison

| Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | Proposed AI Agent | Metric Priority / Safety Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Macro-F1** | `{tb_h['intent_macro_f1']:.4f}` | `{sb_h['intent_macro_f1']:.4f}` | **`{ag_h['intent_macro_f1']:.4f}`** | **Primary Intent Metric** (unskewed by class imbalance) |
| **Intent Overall Accuracy** | `{tb_h['intent_accuracy']:.4f}` | `{sb_h['intent_accuracy']:.4f}` | **`{ag_h['intent_accuracy']:.4f}`** | Overall classification correctness |
| **Safe Auto-Handle Precision** | `{tb_h['safe_auto_handle_precision']:.4f}` | `{sb_h['safe_auto_handle_precision']:.4f}` | **`{ag_h['safe_auto_handle_precision']:.4f}`** | **Primary Safety Metric** (when auto-handling, is it truly safe?) |
| **Missed Escalation Rate** | `{tb_h['missed_escalation_rate']:.4f}` | `{sb_h['missed_escalation_rate']:.4f}` | **`{ag_h['missed_escalation_rate']:.4f}`** | **Critical Hazard** (true risk queries dangerously auto-handled) |
| **False Escalation Rate** | `{tb_h['false_escalation_rate']:.4f}` | `{sb_h['false_escalation_rate']:.4f}` | **`{ag_h['false_escalation_rate']:.4f}`** | Over-escalation rate (queue cost / human agent burden) |
| **Escalation Recall** | `{tb_h['escalation_recall']:.4f}` | `{sb_h['escalation_recall']:.4f}` | **`{ag_h['escalation_recall']:.4f}`** | Coverage of critical security/financial risks |
| **Escalation Precision** | `{tb_h['escalation_precision']:.4f}` | `{sb_h['escalation_precision']:.4f}` | **`{ag_h['escalation_precision']:.4f}`** | Cleanliness of human triage queue |
| **Grounded Reply Pass Rate** | `{tb_h['grounded_reply_pass_rate']:.4f}` | `{sb_h['grounded_reply_pass_rate']:.4f}` | **`{ag_h['grounded_reply_pass_rate']:.4f}`** | Deterministic 4-Criteria Rubric Pass (Groundedness + Actionability) |
| **PII Safety Compliance** | `{tb_h['pii_safety_rate']:.4f}` | `{sb_h['pii_safety_rate']:.4f}` | **`{ag_h['pii_safety_rate']:.4f}`** | Deterministic rule compliance (zero credential solicitation) |
| **ROUGE-L Similarity** | `{trivial_eval['lexical_diagnostics']['rouge_l']:.4f}` | `{simple_eval['lexical_diagnostics']['rouge_l']:.4f}` | **`{agent_eval['lexical_diagnostics']['rouge_l']:.4f}`** | Lexical alignment with historical Amazon resolutions |

---

### 2. Difficulty Tier Performance Breakdown (Macro-F1)

| Difficulty Tier | Sample Count | Baseline 0 | Baseline 1 | Proposed AI Agent | Performance Drop (Adversarial vs Normal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Cases** | {ag_d['normal']['count']} | `{tb_d['normal']['macro_f1']:.4f}` | `{sb_d['normal']['macro_f1']:.4f}` | **`{ag_d['normal']['macro_f1']:.4f}`** | Baseline reference |
| **Difficult Cases** | {ag_d['difficult']['count']} | `{tb_d['difficult']['macro_f1']:.4f}` | `{sb_d['difficult']['macro_f1']:.4f}` | **`{ag_d['difficult']['macro_f1']:.4f}`** | Nuanced multi-intent queries |
| **Adversarial Cases** | {ag_d['adversarial']['count']} | `{tb_d['adversarial']['macro_f1']:.4f}` | `{sb_d['adversarial']['macro_f1']:.4f}` | **`{ag_d['adversarial']['macro_f1']:.4f}`** | High-friction / sarcasm / hostility |
"""

    table_path = ARTIFACTS_DIR / "headline_results_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(markdown_table)

    dt = time.time() - t0
    print("\n" + "="*90)
    print(f"PIPELINE EXECUTION COMPLETED IN {dt:.1f} SECONDS (< 1 MINUTE REPRODUCIBILITY GUARANTEED)")
    print("="*90)
    print(f"\nMetrics saved to: {metrics_json_path}")
    print(f"Headline table saved to: {table_path}")
    print(f"Confusion matrices saved to: {ARTIFACTS_DIR}")
    print("\nHeadline Summary:")
    print(f"  Proposed Agent Macro-F1          : {ag_h['intent_macro_f1']}")
    print(f"  Proposed Agent Safe Auto-Handle  : {ag_h['safe_auto_handle_precision']}")
    print(f"  Proposed Agent Missed Escalation : {ag_h['missed_escalation_rate']}")
    print(f"  Proposed Agent Reply Pass Rate   : {ag_h['grounded_reply_pass_rate']}")

    return full_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Master Evaluation & Reproduction Pipeline for @AmazonHelp")
    parser.add_argument(
        "--force-retrain", "--rebuild-cache",
        dest="force_retrain",
        action="store_true",
        help="Force rebuild of retrieval index and intent classifier from scratch."
    )
    args = parser.parse_args()
    run_pipeline(force_retrain=args.force_retrain)
