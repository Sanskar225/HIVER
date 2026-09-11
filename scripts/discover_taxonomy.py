import json
import re
import collections
import pandas as pd
import pyarrow.parquet as pq

def is_english(text):
    # Simple heuristic to filter non-English tweets
    words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    common_en = {"the", "my", "to", "and", "is", "for", "in", "it", "you", "that", "on", "was", "have", "with", "this", "me", "from", "order", "delivery", "package", "amazon"}
    return len(words.intersection(common_en)) >= 2

def discover_clusters():
    print("Loading conversations...")
    table = pq.read_table("data/raw/twcs_full.parquet", columns=["company", "conversation"])
    df = table.to_pandas()
    df_amz = df[df["company"] == "AmazonHelp"]
    
    # Extract first customer turn from conversations
    cust_messages = []
    for conv in df_amz["conversation"]:
        lines = conv.strip().split("\n")
        first_cust = None
        for line in lines:
            if line.startswith("Customer: "):
                first_cust = line[len("Customer: "):]
                break
        if first_cust and is_english(first_cust) and len(first_cust) >= 25:
            # Clean mentions and URLs for clustering
            cleaned = re.sub(r"@\w+", "", first_cust)
            cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
            cust_messages.append({
                "raw": first_cust,
                "cleaned": cleaned
            })
        if len(cust_messages) >= 2000:
            break

    print(f"Collected {len(cust_messages)} English customer initial queries from @AmazonHelp.")
    
    # Extract 1000 sample for audit
    sample_1000 = cust_messages[:1000]
    
    # Cluster based on lexical and semantic pattern anchors
    cluster_definitions = {
        "DELIVERY_STATUS_DELAY": r"\b(delivery|deliver|late|delayed|delay|where is|tracking|track|carrier|package|parcel|courier|transit|arrive|arriving|eta|status)\b",
        "DAMAGED_WRONG_MISSING": r"\b(damaged|broken|empty box|missing|wrong item|defective|shattered|cracked|ruined|torn|opened|incomplete)\b",
        "REFUND_RETURN_EXCHANGE": r"\b(refund|return|returning|exchange|money back|pickup|drop off|refunded|send back)\b",
        "ORDER_CHANGE_CANCEL": r"\b(cancel|cancelled|cancellation|change address|modify order|wrong address|change payment)\b",
        "ACCOUNT_SECURITY_ACCESS": r"\b(hacked|fraud|unauthorized|phishing|otp|locked out|password|scam|security|compromised|login)\b",
        "SUBSCRIPTION_BILLING_PRIME": r"\b(prime|membership|subscription|charged|fee|billing|debit|card charged|renew|unrecognized charge)\b",
        "TECHNICAL_PRODUCT_SUPPORT": r"\b(kindle|fire stick|echo|alexa|app|website|code|voucher|coupon|promo|error|crash|bug|tv)\b",
    }
    
    cluster_counts = collections.defaultdict(list)
    multi_matched = []
    unmatched = []
    
    for item in sample_1000:
        text = item["cleaned"]
        matches = []
        for name, pat in cluster_definitions.items():
            if re.search(pat, text, re.I):
                matches.append(name)
                
        if len(matches) == 1:
            cluster_counts[matches[0]].append(item["raw"])
        elif len(matches) > 1:
            multi_matched.append((matches, item["raw"]))
            # Assign to first match for counting
            cluster_counts[matches[0]].append(item["raw"])
        else:
            unmatched.append(item["raw"])

    print("\n" + "="*70)
    print("TAXONOMY DISCOVERY FROM 1,000 REAL CUSTOMER MESSAGES")
    print("="*70)
    for name, items in sorted(cluster_counts.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {name:<30}: {len(items):>4} ({len(items)/10:.1f}%)")
    print(f"  MULTI-INTENT COLLISION CANDIDATES: {len(multi_matched):>4} ({len(multi_matched)/10:.1f}%)")
    print(f"  UNMATCHED / GENERAL FEEDBACK     : {len(unmatched):>4} ({len(unmatched)/10:.1f}%)")

    # Save summary and samples to json
    taxonomy_data = {
        "cluster_distribution": {k: len(v) for k, v in cluster_counts.items()},
        "multi_intent_count": len(multi_matched),
        "unmatched_count": len(unmatched),
        "sample_per_intent": {k: v[:5] for k, v in cluster_counts.items()},
        "multi_intent_samples": [
            {"intents": m[0], "text": m[1]} for m in multi_matched[:10]
        ],
        "unmatched_samples": unmatched[:10]
    }
    
    with open("data/taxonomy_discovery.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
    print("\nSaved taxonomy discovery data to data/taxonomy_discovery.json")

if __name__ == "__main__":
    discover_clusters()
