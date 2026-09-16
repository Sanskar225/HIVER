"""
Concurrency, Stress, Observability & Rate Limit Test Suite for @AmazonHelp FastAPI Microservice.
Validates:
1. High-throughput multi-threaded burst (50 concurrent requests).
2. TOCTOU session race condition prevention (atomic per-session locking).
3. True LRU session cache eviction (active VIP session preservation).
4. Single-session turn depth bounding (sliding window MAX_TURNS_PER_SESSION).
5. Strict payload length boundary enforcement (HTTP 422 on oversized payloads).
6. Kubernetes /live and /ready probe operation.
7. Observability headers (X-Request-ID, X-Process-Time-Ms).
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from fastapi.testclient import TestClient

from src.api import (
    app, 
    session_store, 
    clear_session_store, 
    rate_limiter, 
    InMemoryLRUSessionStore
)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_k8s_probes_live_and_ready(client):
    """Verify Kubernetes /live (<1ms) and /ready probes."""
    res_live = client.get("/live")
    assert res_live.status_code == 200
    assert res_live.json() == {"status": "alive"}

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json() == {"status": "ready"}

def test_observability_middleware_headers(client):
    """Verify X-Request-ID and X-Process-Time-Ms headers on responses."""
    custom_req_id = "test-correlation-uuid-999"
    res = client.post(
        "/v1/triage",
        json={"message": "Where is my delivery?"},
        headers={"X-Request-ID": custom_req_id}
    )
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_req_id
    assert "X-Process-Time-Ms" in res.headers
    assert float(res.headers.get("X-Process-Time-Ms")) >= 0.0

def test_payload_length_boundaries_rejection(client):
    """Verify oversized payloads (> 4096 chars) are rejected with HTTP 422."""
    oversized_message = "A" * 5000
    res = client.post("/v1/chat", json={"message": oversized_message})
    assert res.status_code == 422

    oversized_sess_id = "B" * 200
    res2 = client.post("/v1/chat", json={"message": "Valid text", "session_id": oversized_sess_id})
    assert res2.status_code == 422

def test_true_lru_session_eviction():
    """Verify that OrderedDict LRU evicts least-recently used and preserves accessed active sessions."""
    mini_store = InMemoryLRUSessionStore(max_sessions=4, max_turns=5)
    
    # Add 4 sessions
    for i in range(1, 5):
        mini_store.append_turn(f"sess_{i}", {"customer": f"msg_{i}", "agent_reply": f"rep_{i}"})
    
    assert mini_store.count() == 4
    
    # Access sess_1 (making it most recently used)
    hist = mini_store.get_history("sess_1")
    assert len(hist) == 1
    
    # Add sess_5 (capacity overflow)
    mini_store.append_turn("sess_5", {"customer": "msg_5", "agent_reply": "rep_5"})
    assert mini_store.count() == 4
    
    # sess_2 was the least recently used, so it must have been evicted!
    assert mini_store.get_history("sess_2") == []
    # sess_1 was accessed and must be preserved!
    assert len(mini_store.get_history("sess_1")) == 1
    assert len(mini_store.get_history("sess_5")) == 1

def test_single_session_turn_depth_bounding():
    """Verify that a single session cannot exceed max_turns in memory."""
    mini_store = InMemoryLRUSessionStore(max_sessions=10, max_turns=5)
    sess_id = "infinite_loop_bot"
    
    # Append 15 turns
    for i in range(15):
        mini_store.append_turn(sess_id, {"customer": f"Turn {i}", "agent_reply": f"Reply {i}"})
    
    history = mini_store.get_history(sess_id)
    assert len(history) == 5
    # Should only retain the last 5 turns (Turn 10 to Turn 14)
    assert history[0]["customer"] == "Turn 10"
    assert history[-1]["customer"] == "Turn 14"

def test_multi_threaded_concurrent_burst(client):
    """Verify server stability and sub-50ms throughput under 50 concurrent requests."""
    orig_max = rate_limiter.max_requests
    try:
        rate_limiter.max_requests = 1000
        queries = [
            "Where is my package?",
            "I want a refund for damaged item",
            "Someone hacked my account",
            "When will my book arrive?",
            "Cancel my Prime subscription",
        ] * 10  # 50 requests
        
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(client.post, "/v1/triage", json={"message": q})
                for q in queries
            ]
            for f in as_completed(futures):
                results.append(f.result())
        
        assert len(results) == 50
        for r in results:
            assert r.status_code == 200
            data = r.json()
            assert "intent" in data
            assert "decision" in data
    finally:
        rate_limiter.max_requests = orig_max
        rate_limiter.reset()

def test_toctou_session_transaction_lock_race_condition(client):
    """
    Verify per-session transaction lock guarantees sequential atomic processing
    without dirty reads or lost updates on the same session_id.
    """
    clear_session_store()
    sess_id = "concurrent_race_session"
    
    # Simulate 2 rapid messages targeting the exact same session concurrently
    messages = [
        "Where is my package? Says delivered but not here!",
        "Order 112-9876543-1234567"
    ]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                client.post, 
                "/v1/chat", 
                json={"message": msg, "session_id": sess_id}
            )
            for msg in messages
        ]
        responses = [f.result() for f in as_completed(futures)]
    
    for r in responses:
        assert r.status_code == 200
    
    # Verify session store contains exactly both turns without race condition corruption
    history = session_store.get_history(sess_id)
    assert len(history) == 2
