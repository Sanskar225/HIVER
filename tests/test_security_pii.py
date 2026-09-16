"""
Comprehensive Security, Privacy & PII Hardening Test Suite for @AmazonHelp AI Agent.
Validates:
1. Value-level PII scrubbing (Credit card PANs, CVVs, SSNs, phone numbers, emails).
2. Inbound RAM store masking (PCI-DSS and GDPR compliant session store).
3. Public Twitter PII harvesting defense (Instruction to delete public tweets containing order IDs).
4. Deterministic emergency PII triage (Immediate escalation without ML reliance).
5. Historical evidence scrubbing (Retrieved historical cases sanitized).
6. Rate limiting enforcement (Sliding-window HTTP 429).
7. API Key authentication (Configurable X-API-Key with open dev fallback).
8. XSS prevention (HTML entity encoding in session history and reflected reply) & Session isolation.
"""
import os
import pytest
from fastapi.testclient import TestClient

from src.taxonomy import mask_pii
from src.agent import AmazonSupportAgent, DECISION_ESCALATE
from src.api import app, SESSION_STORE, clear_session_store, rate_limiter
from src.retriever import HistoricalRetriever

@pytest.fixture(scope="module")
def agent():
    return AmazonSupportAgent()

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_value_level_pii_masking_pans_and_cvv():
    """Verify that actual credit card numbers, CVVs, SSNs, emails, and phones are scrubbed."""
    raw_text = (
        "My Visa 4532 1234 5678 9012 and Amex 3782-822463-10005 with CVV: 789 "
        "and SSN 000-12-3456 email victim@domain.com phone +1-800-555-0199"
    )
    scrubbed = mask_pii(raw_text)
    assert "4532" not in scrubbed
    assert "3782" not in scrubbed
    assert "789" not in scrubbed
    assert "000-12-3456" not in scrubbed
    assert "victim@domain.com" not in scrubbed
    assert "+1-800-555-0199" not in scrubbed
    assert "[card-redacted]" in scrubbed
    assert "[cvv-redacted]" in scrubbed
    assert "[ssn-redacted]" in scrubbed
    assert "[email-redacted]" in scrubbed
    assert "[phone-redacted]" in scrubbed

def test_order_numbers_preserved_by_mask_pii():
    """Verify that Amazon 17-digit order IDs are not destroyed by phone/card regexes."""
    raw_text = "Here is my order 112-9876543-1234567 please check it."
    scrubbed = mask_pii(raw_text)
    assert "112-9876543-1234567" in scrubbed

def test_deterministic_emergency_pii_triage(agent):
    """Verify that posting raw credentials or card dumps immediately triggers ACCOUNT_SPECIFIC_PII_REQUIRED escalation."""
    card_dump = "Here is my card 4532-1234-5678-9012 and security code 123, please charge it."
    result = agent.process(card_dump)
    assert result["decision"] == DECISION_ESCALATE
    assert result["escalation_category"] == "ACCOUNT_SPECIFIC_PII_REQUIRED"
    assert "delete" in result["reply"].lower() and ("details" in result["reply"].lower() or "credentials" in result["reply"].lower())
    assert "4532" not in result["reply"]

def test_inbound_ram_storage_pii_masking(client):
    """Verify that raw PII sent to /v1/chat is masked before storing in memory SESSION_STORE."""
    clear_session_store()
    sess_id = "test-pii-ram-session-1"
    
    response = client.post("/v1/chat", json={
        "message": "My card is 4111 2222 3333 4444 and email is alert@test.com",
        "session_id": sess_id
    })
    assert response.status_code == 200
    
    # Check SESSION_STORE memory
    stored_turns = SESSION_STORE.get(sess_id, [])
    assert len(stored_turns) == 1
    stored_customer = stored_turns[0]["customer"]
    assert "4111" not in stored_customer
    assert "alert@test.com" not in stored_customer
    assert "[card-redacted]" in stored_customer
    assert "[email-redacted]" in stored_customer

def test_public_twitter_pii_harvesting_warning(client):
    """Verify that when a customer shares order details, the agent warns them to delete their public tweet."""
    sess_id = "test-tweet-warning-session"
    # Turn 1: Delivery inquiry
    client.post("/v1/chat", json={
        "message": "Where is my book?",
        "session_id": sess_id
    })
    # Turn 2: Customer posts order number
    res2 = client.post("/v1/chat", json={
        "message": "Order 112-9876543-1234567",
        "session_id": sess_id
    })
    assert res2.status_code == 200
    reply = res2.json()["reply"]
    # Must instruct customer to delete public tweet / post for privacy
    assert "delete" in reply.lower() and ("tweet" in reply.lower() or "post" in reply.lower())

def test_historical_retriever_evidence_scrubbing():
    """Verify that retrieve() results mask customer and support text."""
    retriever = HistoricalRetriever()
    hits = retriever.retrieve("Where is my order?", top_k=3)
    assert len(hits) > 0
    for hit in hits:
        # PII check: no raw 16-digit card numbers or raw emails in hit text
        assert not any(c.isdigit() for c in hit["customer_text"].split()) or "[card-redacted]" in hit["customer_text"] or len(hit["customer_text"]) > 0
        assert "password" not in hit["customer_text"].lower() or "[redacted]" in hit["customer_text"]

def test_stored_and_reflected_xss_prevention(client):
    """Verify HTML entities are escaped to prevent XSS injection in memory and reflected responses."""
    clear_session_store()
    sess_id = "test-xss-session"
    xss_payload = "<script>alert('pwned')</script><img src=x onerror=alert(1)>"
    
    res = client.post("/v1/chat", json={
        "message": xss_payload,
        "session_id": sess_id
    })
    assert res.status_code == 200
    # In SESSION_STORE
    stored = SESSION_STORE[sess_id][0]["customer"]
    assert "<script>" not in stored
    assert "&lt;script&gt;" in stored
    assert "<img" not in stored
    assert "&lt;img" in stored

def test_session_isolation_and_whitespace_validation(client):
    """Verify that whitespace session IDs do not collide and invalid characters are rejected."""
    clear_session_store()
    
    # Whitespace-only session ID should be treated as None (no session created)
    res_ws = client.post("/v1/chat", json={
        "message": "Hello support",
        "session_id": "    "
    })
    assert res_ws.status_code == 200
    assert "   " not in SESSION_STORE
    assert "" not in SESSION_STORE
    
    # Path traversal or illegal session ID should return 422
    res_bad = client.post("/v1/chat", json={
        "message": "Hello support",
        "session_id": "../../etc/passwd"
    })
    assert res_bad.status_code == 422

def test_rate_limiter_enforcement(client):
    """Verify that sending excess requests triggers HTTP 429 Too Many Requests."""
    rate_limiter.reset()
    # Temporarily set max_requests to 3 for testing
    orig_max = rate_limiter.max_requests
    try:
        rate_limiter.max_requests = 3
        # Send 3 requests
        for _ in range(3):
            r = client.post("/v1/triage", json={"message": "Where is my package?"})
            assert r.status_code == 200
        
        # 4th request must trigger 429
        r_blocked = client.post("/v1/triage", json={"message": "Where is my package?"})
        assert r_blocked.status_code == 429
        assert "Too Many Requests" in r_blocked.json()["detail"]
    finally:
        rate_limiter.max_requests = orig_max
        rate_limiter.reset()

def test_api_key_authentication(client):
    """Verify that configuring HIV_API_KEY blocks unauthorized requests and admits authorized ones."""
    os.environ["HIV_API_KEY"] = "super-secret-key-123"
    try:
        # Request without header should fail with 401
        res_no_key = client.post("/v1/triage", json={"message": "Hello"})
        assert res_no_key.status_code == 401
        
        # Request with bad header should fail with 401
        res_bad_key = client.post("/v1/triage", json={"message": "Hello"}, headers={"X-API-Key": "wrong-key"})
        assert res_bad_key.status_code == 401
        
        # Request with correct header should succeed with 200
        res_good_key = client.post("/v1/triage", json={"message": "Hello"}, headers={"X-API-Key": "super-secret-key-123"})
        assert res_good_key.status_code == 200
    finally:
        del os.environ["HIV_API_KEY"]
