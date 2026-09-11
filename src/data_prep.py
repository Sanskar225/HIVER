import os
import re
import random
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_BRAND,
    RANDOM_SEED
)

def is_english_tweet(text: str) -> bool:
    if not text or len(text) < 15:
        return False
    # English stopword heuristic
    common_en = {
        "the", "my", "to", "and", "is", "for", "in", "it", "you", "that", "on", 
        "was", "have", "with", "this", "me", "from", "order", "delivery", "package", 
        "amazon", "help", "not", "can", "please", "been", "when", "why", "what", "where"
    }
    words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    return len(words.intersection(common_en)) >= 2

def parse_conversation_turns(conv_text: str):
    lines = conv_text.strip().split("\n")
    cust_first = None
    supp_first = None
    turns = []
    
    for line in lines:
        if line.startswith("Customer: "):
            content = line[len("Customer: "):].strip()
            turns.append(("Customer", content))
            if cust_first is None:
                cust_first = content
        elif line.startswith("Support: "):
            content = line[len("Support: "):].strip()
            turns.append(("Support", content))
            if supp_first is None:
                supp_first = content
                
    return cust_first, supp_first, turns

def clean_twitter_noise(text: str) -> str:
    # Normalize anonymized numbers like @115821 to @User
    text = re.sub(r"@\d+", "@User", text)
    # Remove t.co shortened links for clean retrieval
    text = re.sub(r"https?://t\.co/\S+", "[link]", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_data_prep(sample_golden_pool_size: int = 1200):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[Phase 1] Loading full dataset from {RAW_DATA_PATH}...")
    table = pq.read_table(RAW_DATA_PATH)
    df = table.to_pandas()
    print(f"Total rows in raw parquet: {len(df):,}")
    
    # Filter for target brand
    df_brand = df[df["company"] == TARGET_BRAND].copy()
    print(f"Total conversations for @{TARGET_BRAND}: {len(df_brand):,}")
    
    # Parse conversations into structured pairs
    parsed_records = []
    for _, row in df_brand.iterrows():
        cid = row["conversation_id"]
        conv_text = row["conversation"]
        c_first, s_first, turns = parse_conversation_turns(conv_text)
        
        # We need a valid customer turn and support reply
        if c_first and s_first and is_english_tweet(c_first):
            c_clean = clean_twitter_noise(c_first)
            s_clean = clean_twitter_noise(s_first)
            
            parsed_records.append({
                "conversation_id": cid,
                "customer_text": c_clean,
                "support_reply": s_clean,
                "raw_customer_text": c_first,
                "raw_support_reply": s_first,
                "turn_count": len(turns)
            })
            
    df_clean = pd.DataFrame(parsed_records)
    # Deduplicate on customer text to eliminate exact duplicate spam/bot tweets
    df_clean = df_clean.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    print(f"Clean, unique English customer->support pairs: {len(df_clean):,}")
    
    # Partition into strictly disjoint pools:
    # 1. Golden Candidate Pool (held out from retrieval KB)
    # 2. Historical Knowledge Base Corpus
    rng = random.Random(RANDOM_SEED)
    all_indices = list(range(len(df_clean)))
    rng.shuffle(all_indices)
    
    golden_indices = all_indices[:sample_golden_pool_size]
    kb_indices = all_indices[sample_golden_pool_size:]
    
    df_golden_pool = df_clean.iloc[golden_indices].reset_index(drop=True)
    df_kb = df_clean.iloc[kb_indices].reset_index(drop=True)
    
    # Verify zero leakage between Golden Pool and Historical KB
    golden_cids = set(df_golden_pool["conversation_id"])
    kb_cids = set(df_kb["conversation_id"])
    assert len(golden_cids.intersection(kb_cids)) == 0, "Data leakage detected between Golden Pool and KB!"
    print("[Verification Passed] Zero conversation ID overlap between Golden Pool and Knowledge Base.")
    
    # Save processed files
    golden_pool_path = PROCESSED_DATA_DIR / "golden_candidate_pool.parquet"
    kb_path = PROCESSED_DATA_DIR / "kb_corpus.parquet"
    
    df_golden_pool.to_parquet(golden_pool_path, index=False)
    df_kb.to_parquet(kb_path, index=False)
    
    print(f"Saved Golden Candidate Pool ({len(df_golden_pool):,} rows) to {golden_pool_path}")
    print(f"Saved Historical Knowledge Base ({len(df_kb):,} rows) to {kb_path}")
    
    return df_golden_pool, df_kb

if __name__ == "__main__":
    run_data_prep()
