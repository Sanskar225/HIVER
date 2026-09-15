import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture(scope='module')
def client():
    return TestClient(app)

def test_api_health_endpoint(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'healthy'
    assert data['brand'] == 'AmazonHelp'

def test_api_triage_endpoint(client):
    res = client.post('/v1/triage', json={'message': 'When will my package be delivered?'})
    assert res.status_code == 200
    data = res.json()
    assert data['decision'] == 'AUTO_HANDLE'

def test_api_chat_stateful_multi_turn(client):
    # Turn 1: Stolen package
    res1 = client.post('/v1/chat', json={
        'message': 'Package says delivered but was never left on my porch!',
        'session_id': 'test-api-sess-1'
    })
    assert res1.status_code == 200
    assert res1.json()['decision'] == 'ESCALATE'
    
    # Turn 2: Provide order number
    res2 = client.post('/v1/chat', json={
        'message': 'Order # 112-1234567-9876543',
        'session_id': 'test-api-sess-1'
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2['multi_turn'] is True
    assert data2['decision'] == 'ESCALATE'
