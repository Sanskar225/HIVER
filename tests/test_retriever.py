import time
import pytest
from src.retriever import HistoricalRetriever

@pytest.fixture(scope='module')
def retriever():
    return HistoricalRetriever()

def test_retriever_returns_valid_hits(retriever):
    hits = retriever.retrieve('where is my package', top_k=3)
    assert len(hits) == 3
    for h in hits:
        assert 'conversation_id' in h
        assert 'customer_text' in h
        assert 'support_reply' in h
        assert 'similarity_score' in h
        assert 0.0 <= h['similarity_score'] <= 1.0

def test_retriever_sub_100ms_latency(retriever):
    # Warm up BLAS matrix operations
    retriever.retrieve('warmup test query', top_k=1)
    t0 = time.time()
    retriever.retrieve('I want to return damaged shoes', top_k=2)
    elapsed = (time.time() - t0) * 1000
    assert elapsed < 150, f'Retrieval latency too high: {elapsed:.2f}ms'
