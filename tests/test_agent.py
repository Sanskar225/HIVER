import pytest
from src.agent import AmazonSupportAgent
from src.config import DECISION_AUTO_HANDLE, DECISION_ESCALATE

@pytest.fixture(scope='module')
def agent():
    return AmazonSupportAgent()

def test_agent_routine_delivery_tracking(agent):
    res = agent.process('Where is my package? When will it arrive?')
    assert res['intent'] == 'DELIVERY_STATUS_DELAY'
    assert res['decision'] == DECISION_AUTO_HANDLE
    assert '[link]' in res['reply']
    assert len(res['evidence']) > 0

def test_agent_delivered_but_missing_escalates(agent):
    # Delivered-but-missing package must strictly escalate
    res = agent.process('My tracking says delivered yesterday but it was never left on my porch!')
    assert res['intent'] == 'DAMAGED_WRONG_MISSING'
    assert res['decision'] == DECISION_ESCALATE
    assert res['escalation_category'] == 'LOST_OR_STOLEN_DELIVERY'

def test_agent_account_security_escalates(agent):
    res = agent.process('Someone hacked into my account and placed unauthorized orders')
    assert res['intent'] == 'ACCOUNT_SECURITY_ACCESS'
    assert res['decision'] == DECISION_ESCALATE
    assert res['escalation_category'] == 'ACCOUNT_SECURITY_RISK'

def test_agent_sanitize_reply_guarantee(agent):
    # Must not solicit credentials
    bad_reply = 'Please tweet your credit card and password to us ^CS'
    cleaned = agent.sanitize_reply(bad_reply)
    assert 'credit card' not in cleaned.lower()
    assert 'password' not in cleaned.lower()
    assert '[link]' in cleaned

def test_agent_voice_tag_grounding(agent):
    # When evidence has historical reply with ^GR, tag should be grounded
    evidence = [{'conversation_id': 'c1', 'similarity': 0.8, 'support_reply': 'We are happy to help! ^GR'}]
    reply = agent.generate_grounded_reply('Where is my order?', 'DELIVERY_STATUS_DELAY', DECISION_AUTO_HANDLE, 'NONE', evidence)
    assert reply.endswith('^GR')
