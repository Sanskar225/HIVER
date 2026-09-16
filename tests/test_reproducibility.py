"""
Reproducibility, Self-Healing Deserializer, and CLI Verification Test Suite.
Validates:
1. HistoricalRetriever force_rebuild rebuilds TF-IDF index from scratch.
2. AmazonSupportAgent force_retrain retrains stratified model from scratch.
3. Self-healing fallback gracefully recovers when pickled artifacts are corrupted or incompatible.
4. Dataset assets (kb_corpus.parquet, golden_eval_set.json) exist and have valid shapes.
"""
import pickle
import pytest
from pathlib import Path
from src.retriever import HistoricalRetriever, INDEX_CACHE_PATH
from src.agent import AmazonSupportAgent, AGENT_MODEL_CACHE_PATH
from src.config import PROCESSED_DATA_DIR, GOLDEN_DATA_DIR

def test_processed_corpus_and_golden_dataset_integrity():
    """Verify that kb_corpus.parquet and golden_eval_set.json are tracked and valid."""
    kb_path = PROCESSED_DATA_DIR / "kb_corpus.parquet"
    golden_path = GOLDEN_DATA_DIR / "golden_eval_set.json"

    assert kb_path.exists(), f"Corpus missing: {kb_path}"
    assert golden_path.exists(), f"Golden set missing: {golden_path}"

    import pandas as pd
    df_kb = pd.read_parquet(kb_path)
    assert len(df_kb) > 50000, f"Expected 50k+ cases, got {len(df_kb)}"
    assert "customer_text" in df_kb.columns
    assert "support_reply" in df_kb.columns

    df_gold = pd.read_json(golden_path)
    assert len(df_gold) == 200, f"Expected 200 golden cases, got {len(df_gold)}"

def test_retriever_force_rebuild():
    """Verify that force_rebuild=True reconstructs index from scratch without error."""
    retriever = HistoricalRetriever(force_rebuild=True)
    assert retriever.df_kb is not None
    assert retriever.vectorizer is not None
    assert retriever.tfidf_matrix is not None
    assert retriever.tfidf_matrix.shape[0] == len(retriever.df_kb)
    
    # Test retrieval functionality
    hits = retriever.retrieve("Where is my package?", top_k=2)
    assert len(hits) == 2
    assert "customer_text" in hits[0]

def test_agent_force_retrain():
    """Verify that force_retrain=True retrains the calibrated classifier from scratch."""
    agent = AmazonSupportAgent(force_retrain=True)
    assert agent.vectorizer is not None
    assert agent.classifier is not None
    
    # Test prediction functionality
    result = agent.process("When will my delivery arrive?")
    assert result["intent"] == "DELIVERY_STATUS_DELAY"
    assert result["intent_confidence"] > 0.5
    assert result["decision"] == "AUTO_HANDLE"

def test_retriever_self_healing_corrupted_pickle(tmp_path):
    """Verify that corrupted or cross-version incompatible pickle files trigger graceful self-healing."""
    # Write corrupt data to a temporary cache file
    fake_cache = tmp_path / "corrupted_index.pkl"
    fake_cache.write_bytes(b"INVALID_PICKLE_OPCODE_CORRUPTED_STREAM")
    
    # Patch the cache path temporarily to verify self-healing
    import src.retriever as ret_mod
    orig_path = ret_mod.INDEX_CACHE_PATH
    try:
        ret_mod.INDEX_CACHE_PATH = fake_cache
        # Should not raise UnpicklingError; should self-heal and rebuild
        retriever = HistoricalRetriever(force_rebuild=False)
        assert retriever.df_kb is not None
        assert retriever.vectorizer is not None
    finally:
        ret_mod.INDEX_CACHE_PATH = orig_path

def test_agent_self_healing_corrupted_pickle(tmp_path):
    """Verify that corrupted agent model pickle triggers graceful self-healing retrain."""
    fake_cache = tmp_path / "corrupted_agent_model.pkl"
    fake_cache.write_bytes(b"NOT_A_VALID_PICKLE")
    
    import src.agent as agent_mod
    orig_path = agent_mod.AGENT_MODEL_CACHE_PATH
    try:
        agent_mod.AGENT_MODEL_CACHE_PATH = fake_cache
        # Should not raise UnpicklingError; should self-heal and retrain
        agent = AmazonSupportAgent(force_retrain=False)
        assert agent.vectorizer is not None
        assert agent.classifier is not None
    finally:
        agent_mod.AGENT_MODEL_CACHE_PATH = orig_path
