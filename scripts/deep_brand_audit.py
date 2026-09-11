import os
import re
import math
import json
import collections
import pandas as pd
import pyarrow.parquet as pq

PARQUET_PATH = "data/raw/twcs_full.parquet"

CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "TMobileHelp"
]

WEIGHTS = {
    "usable_conversations": 0.25,
    "resolution_coverage": 0.20,
    "issue_diversity": 0.20,
    "thread_richness": 0.15,
    "data_quality": 0.10,
    "evaluation_difficulty": 0.10
}

def parse_turns(conversation_text):
    lines = conversation_text.strip().split("\n")
    turns = []
    current_speaker = None
    current_text = []
    for line in lines:
        if line.startswith("Customer: "):
            if current_speaker:
                turns.append((current_speaker, " ".join(current_text).strip()))
            current_speaker = "Customer"
            current_text = [line[len("Customer: "):]]
        elif line.startswith("Support: "):
            if current_speaker:
                turns.append((current_speaker, " ".join(current_text).strip()))
            current_speaker = "Support"
            current_text = [line[len("Support: "):]]
        else:
            current_text.append(line)
    if current_speaker and current_text:
        turns.append((current_speaker, " ".join(current_text).strip()))
    return turns

def analyze_brand(df_brand, brand_name, sample_n=5000):
    total_convs = len(df_brand)
    sample_df = df_brand.sample(min(sample_n, total_convs), random_state=42)
    
    parsed_samples = []
    first_inbound_texts = []
    first_reply_texts = []
    turn_counts = []
    multi_turn_count = 0
    has_resolution_count = 0
    dm_only_count = 0
    
    # Actionable resolution indicators (brand gives steps, links, or specific policy answers)
    resolution_keywords = re.compile(r"link|refund|track|cancel|setting|update|restart|check|step|order|form|app|visit|call", re.I)
    dm_pattern = re.compile(r"dm|direct message|private message", re.I)
    
    # Escalation indicators (anger, loss, theft, danger, legal, fraud, money)
    risk_keywords = re.compile(r"stolen|lost|refund|money|charge|fraud|hacked|broken|damage|scam|lawyer|police|court|unacceptable|furious|never arrived", re.I)
    high_risk_count = 0
    
    # Quality indicators
    short_junk_count = 0
    total_cust_len = 0
    
    all_tokens = []
    
    for _, row in sample_df.iterrows():
        turns = parse_turns(row["conversation"])
        parsed_samples.append(turns)
        turn_counts.append(len(turns))
        if len(turns) >= 3:
            multi_turn_count += 1
            
        # First customer turn
        cust_turns = [t[1] for t in turns if t[0] == "Customer"]
        supp_turns = [t[1] for t in turns if t[0] == "Support"]
        
        if cust_turns:
            c_text = cust_turns[0]
            first_inbound_texts.append(c_text)
            total_cust_len += len(c_text)
            if len(c_text) < 20:
                short_junk_count += 1
            if risk_keywords.search(c_text):
                high_risk_count += 1
                
            # Tokenize for diversity
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", c_text)]
            all_tokens.extend(words)
            
        if supp_turns:
            s_text = supp_turns[0]
            first_reply_texts.append(s_text)
            if resolution_keywords.search(s_text):
                has_resolution_count += 1
            if dm_pattern.search(s_text) and not resolution_keywords.search(s_text):
                dm_only_count += 1

    n_samples = len(sample_df)
    
    # 1. Usable conversations metric (raw scale: 0 - 100 normalized against max candidate)
    usable_conv_raw = total_convs
    
    # 2. Resolution coverage: % of first replies that contain concrete guidance / resolution
    resolution_cov_pct = (has_resolution_count / n_samples) * 100
    
    # 3. Issue Diversity: Vocabulary entropy & vocabulary size
    token_counts = collections.Counter(all_tokens)
    total_tokens = sum(token_counts.values())
    vocab_entropy = -sum((c / total_tokens) * math.log2(c / total_tokens) for c in token_counts.values()) if total_tokens > 0 else 0
    vocab_size = len(token_counts)
    
    # 4. Thread Richness: % of conversations with >= 3 turns, and average turns
    multi_turn_pct = (multi_turn_count / n_samples) * 100
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    
    # 5. Data Quality: 100 - % short junk, and average customer query length
    junk_pct = (short_junk_count / n_samples) * 100
    avg_cust_len = total_cust_len / n_samples
    data_quality_score = max(0, 100 - junk_pct * 1.5)
    
    # 6. Evaluation Difficulty: High-risk & nuanced escalation rate (ideal ~20-40% so classes are non-trivial)
    high_risk_pct = (high_risk_count / n_samples) * 100
    # Difficulty is optimal when there is strong balance between safe auto-handle and high-risk escalation
    # A distribution of 30% high-risk / 70% routine is ideal for evaluating triage safety
    diff_balance_score = 100 - abs(high_risk_pct - 35) * 2.0

    # Collect 3 real representative conversations
    rep_convs = []
    for turns in parsed_samples[:15]:
        if len(turns) >= 2 and len(rep_convs) < 3:
            rep_convs.append(turns)

    return {
        "brand": brand_name,
        "total_conversations": total_convs,
        "sample_size": n_samples,
        "resolution_cov_pct": round(resolution_cov_pct, 2),
        "dm_only_pct": round((dm_only_count / n_samples) * 100, 2),
        "vocab_entropy": round(vocab_entropy, 2),
        "vocab_size": vocab_size,
        "multi_turn_pct": round(multi_turn_pct, 2),
        "avg_turns": round(avg_turns, 2),
        "avg_cust_char_len": round(avg_cust_len, 1),
        "junk_pct": round(junk_pct, 2),
        "data_quality_score": round(data_quality_score, 1),
        "high_risk_escalation_pct": round(high_risk_pct, 1),
        "diff_balance_score": round(diff_balance_score, 1),
        "representative_conversations": rep_convs
    }

def main():
    print("Reading full conversations table...")
    table = pq.read_table(PARQUET_PATH, columns=["company", "conversation"])
    df = table.to_pandas()
    print("Loaded table into pandas DataFrame.")

    results = []
    for brand in CANDIDATES:
        df_b = df[df["company"] == brand]
        print(f"Auditing @{brand} ({len(df_b):,} conversations)...")
        res = analyze_brand(df_b, brand)
        results.append(res)

    # Normalize metrics to 0-100 scales for weighted composite scoring
    max_convs = max(r["total_conversations"] for r in results)
    max_vocab_ent = max(r["vocab_entropy"] for r in results)
    min_vocab_ent = min(r["vocab_entropy"] for r in results)
    max_multi = max(r["multi_turn_pct"] for r in results)
    min_multi = min(r["multi_turn_pct"] for r in results)

    scored_results = []
    for r in results:
        # 1. Usable conversations (log-scale normalized to avoid 80k vs 25k blowing out everything)
        # Log-scale gives credit for scale while acknowledging diminishing returns after 20k
        norm_conv = (math.log(r["total_conversations"]) / math.log(max_convs)) * 100
        
        # 2. Resolution coverage
        norm_resol = r["resolution_cov_pct"]
        
        # 3. Issue diversity (entropy normalized)
        norm_div = ((r["vocab_entropy"] - min_vocab_ent) / (max_vocab_ent - min_vocab_ent + 1e-5)) * 40 + 60
        
        # 4. Thread richness
        norm_thread = ((r["multi_turn_pct"] - min_multi) / (max_multi - min_multi + 1e-5)) * 40 + 60
        
        # 5. Data quality
        norm_qual = r["data_quality_score"]
        
        # 6. Evaluation difficulty
        norm_diff = r["diff_balance_score"]
        
        composite_score = (
            WEIGHTS["usable_conversations"] * norm_conv +
            WEIGHTS["resolution_coverage"] * norm_resol +
            WEIGHTS["issue_diversity"] * norm_div +
            WEIGHTS["thread_richness"] * norm_thread +
            WEIGHTS["data_quality"] * norm_qual +
            WEIGHTS["evaluation_difficulty"] * norm_diff
        )
        
        r_copy = dict(r)
        r_copy["scores"] = {
            "usable_conv_score": round(norm_conv, 1),
            "resolution_cov_score": round(norm_resol, 1),
            "diversity_score": round(norm_div, 1),
            "thread_richness_score": round(norm_thread, 1),
            "data_quality_score": round(norm_qual, 1),
            "eval_difficulty_score": round(norm_diff, 1),
            "composite_score": round(composite_score, 2)
        }
        scored_results.append(r_copy)

    # Sort by composite score
    scored_results.sort(key=lambda x: x["scores"]["composite_score"], reverse=True)

    with open("data/brand_audit_results.json", "w") as f:
        json.dump(scored_results, f, indent=2)

    print("\n" + "="*95)
    print(f"{'BRAND':<16} | {'CONVS':<8} | {'RESOL%':<7} | {'ENTROPY':<7} | {'MULTI%':<7} | {'QUAL':<6} | {'DIFF':<6} | {'COMPOSITE SCORE':<15}")
    print("="*95)
    for r in scored_results:
        s = r["scores"]
        print(f"{r['brand']:<16} | {r['total_conversations']:<8,} | {r['resolution_cov_pct']:<7.1f} | {r['vocab_entropy']:<7.2f} | {r['multi_turn_pct']:<7.1f} | {s['data_quality_score']:<6.1f} | {s['eval_difficulty_score']:<6.1f} | {s['composite_score']:<15.2f}")
    print("="*95)

if __name__ == "__main__":
    main()
