import os
import sys
import math
import collections
import pandas as pd
import pyarrow.parquet as pq

def run_audit(parquet_path="data/raw/twcs_full.parquet"):
    print(f"Loading dataset from {parquet_path}...")
    table = pq.read_table(parquet_path)
    print(f"Total rows in dataset: {table.num_rows:,}")
    print("Schema columns:", table.schema.names)

    # Convert to pandas dataframe (or select relevant columns to save memory)
    df = table.select(["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]).to_pandas()
    print("Dataset converted to DataFrame successfully.")

    # 1. Identify all brands (brands have inbound == False)
    brand_counts = df[df["inbound"] == False]["author_id"].value_counts()
    print("\nTop 15 brands by outbound tweet volume:")
    for brand, count in brand_counts.head(15).items():
        print(f"  @{brand:<20}: {count:,} tweets")

    # Select top candidate brands across distinct industries
    candidate_brands = ["AmazonHelp", "AppleSupport", "SpotifyCares", "Uber_Support", "Delta", "sprintcare"]
    
    # Analyze each candidate brand
    print("\n" + "="*80)
    print("AUDITING TOP CANDIDATE BRANDS ACROSS 6 EVALUATION DIMENSIONS")
    print("="*80)

    brand_metrics = {}

    for brand in candidate_brands:
        print(f"\nAnalyzing @{brand}...")
        
        # Outbound tweets from this brand
        brand_outbound = df[(df["inbound"] == False) & (df["author_id"] == brand)]
        n_outbound = len(brand_outbound)
        
        # Customer tweets directed to this brand or replied to by this brand
        # Customer tweets that this brand responded to:
        # brand_outbound['in_response_to_tweet_id'] points to customer tweet_id
        replied_cust_tweet_ids = brand_outbound["in_response_to_tweet_id"].dropna().astype(str).unique()
        
        # Inbound tweets matching these IDs or mentioning the brand
        cust_tweets_replied = df[(df["tweet_id"].astype(str).isin(replied_cust_tweet_ids)) & (df["inbound"] == True)]
        
        # Also inbound tweets mentioning @brand
        mention_pattern = f"@{brand}"
        cust_mentions = df[(df["inbound"] == True) & (df["text"].str.contains(mention_pattern, case=False, na=False))]
        
        # Merge unique customer inbound queries
        all_cust_ids = set(cust_tweets_replied["tweet_id"].astype(str)).union(set(cust_mentions["tweet_id"].astype(str)))
        n_usable_cust_tweets = len(all_cust_ids)
        
        # Merge customer-brand pairs (customer tweet -> brand response)
        # Match brand_outbound on in_response_to_tweet_id == cust_tweet_id
        pairs = brand_outbound.merge(
            df[["tweet_id", "text"]],
            left_on="in_response_to_tweet_id",
            right_on="tweet_id",
            suffixes=("_brand", "_cust")
        )
        n_usable_pairs = len(pairs)
        
        # 2. Resolution Coverage & Density:
        # Check average reply length, concrete guidance vs generic redirects
        avg_brand_reply_len = pairs["text_brand"].str.len().mean() if len(pairs) > 0 else 0
        dm_redirect_rate = pairs["text_brand"].str.contains(r"DM|direct message|private message", case=False, na=False).mean() if len(pairs) > 0 else 0
        actionable_keywords = r"refund|cancel|track|order|link|troubleshoot|update|settings|restart|step"
        actionable_rate = pairs["text_brand"].str.contains(actionable_keywords, case=False, na=False).mean() if len(pairs) > 0 else 0
        
        # 3. Issue Diversity (Vocabulary Entropy on customer tweets)
        cust_sample = cust_mentions["text"].dropna().head(5000)
        tokens = [w.lower() for t in cust_sample for w in t.split() if len(w) > 3 and not w.startswith("@") and not w.startswith("http")]
        token_counts = collections.Counter(tokens)
        total_tokens = sum(token_counts.values())
        entropy = -sum((c / total_tokens) * math.log2(c / total_tokens) for c in token_counts.values()) if total_tokens > 0 else 0
        vocab_size = len(token_counts)
        
        # 4. Conversation / Thread Richness
        # Multi-turn check: how many customer tweets have a response_tweet_id that itself has a response
        has_multi_responses = pairs["response_tweet_id_brand"].notna().sum()
        multi_turn_pct = (has_multi_responses / n_usable_pairs * 100) if n_usable_pairs > 0 else 0
        
        # 5. Data Quality:
        # % of non-empty English text, avg customer query length, rate of intelligible text
        avg_cust_len = cust_sample.str.len().mean()
        short_junk_rate = (cust_sample.str.len() < 20).mean()
        
        # 6. Evaluation Difficulty:
        # Balance between clear FAQs vs high-stakes escalation (refunds, fraud, lost items, hostility)
        escalation_signals = r"stolen|fraud|charge|money|never arrived|broken|hacked|terrible|worst|lawyer|police|lawsuit|unacceptable"
        high_risk_pct = cust_sample.str.contains(escalation_signals, case=False, na=False).mean() * 100
        
        brand_metrics[brand] = {
            "outbound_tweets": n_outbound,
            "usable_customer_tweets": n_usable_cust_tweets,
            "usable_pairs": n_usable_pairs,
            "avg_reply_length": round(avg_brand_reply_len, 1),
            "dm_redirect_rate": round(dm_redirect_rate * 100, 1),
            "actionable_reply_rate": round(actionable_rate * 100, 1),
            "vocab_entropy": round(entropy, 2),
            "vocab_size": vocab_size,
            "multi_turn_pct": round(multi_turn_pct, 1),
            "avg_cust_len": round(avg_cust_len, 1),
            "short_junk_pct": round(short_junk_rate * 100, 1),
            "high_risk_escalation_pct": round(high_risk_pct, 1),
            "sample_pairs": pairs[["text_cust", "text_brand"]].head(5).to_dict(orient="records")
        }

    return brand_metrics

if __name__ == "__main__":
    metrics = run_audit()
