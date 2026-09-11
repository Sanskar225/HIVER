import json
import pandas as pd
import pytest
from src.config import GOLDEN_DATA_DIR, PROCESSED_DATA_DIR, INTENTS

def test_zero_conversation_id_leakage():
    golden_df = pd.read_json(GOLDEN_DATA_DIR / 'golden_eval_set.json')
    kb_df = pd.read_parquet(PROCESSED_DATA_DIR / 'kb_corpus.parquet')
    
    golden_cids = set(golden_df['conversation_id'])
    kb_cids = set(kb_df['conversation_id'])
    
    overlap = golden_cids.intersection(kb_cids)
    assert len(overlap) == 0, f'Data leakage detected! Overlapping conversation IDs: {overlap}'

def test_zero_identical_customer_text_leakage():
    golden_df = pd.read_json(GOLDEN_DATA_DIR / 'golden_eval_set.json')
    kb_df = pd.read_parquet(PROCESSED_DATA_DIR / 'kb_corpus.parquet')
    
    golden_texts = set(golden_df['customer_text'].str.strip().str.lower())
    kb_texts = set(kb_df['customer_text'].str.strip().str.lower())
    
    overlap = golden_texts.intersection(kb_texts)
    assert len(overlap) == 0, f'Exact customer query text leakage detected: {overlap}'

def test_golden_set_stratification_completeness():
    golden_df = pd.read_json(GOLDEN_DATA_DIR / 'golden_eval_set.json')
    assert len(golden_df) == 200
    
    intents_present = set(golden_df['golden_intent'].unique())
    assert intents_present == set(INTENTS)
    
    tier_counts = golden_df['difficulty_tier'].value_counts().to_dict()
    assert tier_counts['normal'] == 140
    assert tier_counts['difficult'] == 40
    assert tier_counts['adversarial'] == 20
