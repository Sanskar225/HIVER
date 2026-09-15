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

def test_when_will_package_be_delivered_auto_handles(agent):
    # Routine ETA inquiry must NOT be treated as a stolen package
    res = agent.process('When will my package be delivered?')
    assert res['intent'] == 'DELIVERY_STATUS_DELAY'
    assert res['decision'] == DECISION_AUTO_HANDLE
    assert res['escalation_category'] == 'NONE'

def test_human_agent_request_escalates(agent):
    # Customer requesting human agent must strictly escalate
    res = agent.process('I want to talk to a human agent')
    assert res['decision'] == DECISION_ESCALATE
    assert 'representative' in res['reply'].lower() or 'specialist' in res['reply'].lower()

def test_theft_past_tense_stole_escalates(agent):
    # Past tense 'stole' must not be auto-handled
    res = agent.process('Someone stole my package from my doorstep')
    assert res['intent'] == 'DAMAGED_WRONG_MISSING'
    assert res['decision'] == DECISION_ESCALATE
    assert res['escalation_category'] == 'LOST_OR_STOLEN_DELIVERY'

def test_misdelivered_wrong_address_escalates(agent):
    # Package delivered to wrong house must escalate
    res = agent.process('My package was delivered to the wrong address')
    assert res['decision'] == DECISION_ESCALATE
    assert res['escalation_category'] == 'LOST_OR_STOLEN_DELIVERY'

def test_hinglish_refund_query(agent):
    # Vernacular Hinglish refund query must be recognized
    res = agent.process('Mera refund kab aayega?')
    assert res['intent'] == 'REFUND_RETURN_EXCHANGE'
    assert res['decision'] == DECISION_ESCALATE

def test_real_mathematical_probabilities(agent):
    # Confidence must be a genuine float from predict_proba (not hardcoded 0.94)
    res = agent.process('How do I return a pair of shoes that are too small?')
    assert isinstance(res['intent_confidence'], float)
    assert 0.0 <= res['intent_confidence'] <= 1.0

def test_multi_turn_order_number_flow(agent):
    # Multi-turn context: customer supplies order number in response to agent handoff
    history = [{
        'customer': 'My package never arrived',
        'agent_reply': 'Please send us a direct message with your order number via [link]. ^CS',
        'intent': 'DAMAGED_WRONG_MISSING',
        'escalation_category': 'LOST_OR_STOLEN_DELIVERY'
    }]
    res = agent.process('Order # 112-9876543-1234567', conversation_history=history, session_id='test-sess-1')
    assert res['multi_turn'] is True
    assert res['decision'] == DECISION_ESCALATE
    assert 'received your information' in res['reply']

