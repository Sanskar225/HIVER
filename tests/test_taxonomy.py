import pytest
from src.config import INTENTS, INTENT_PRIORITY
from src.taxonomy import resolve_intent_collision, evaluate_deterministic_risk

def test_intent_taxonomy_completeness():
    assert len(INTENTS) == 8
    assert len(set(INTENTS)) == 8
    assert 'FEEDBACK_COMPLAINT_GENERAL' in INTENTS
    assert 'DAMAGED_WRONG_MISSING' in INTENTS
    assert 'ACCOUNT_SECURITY_ACCESS' in INTENTS

def test_priority_hierarchy_ordering():
    assert INTENT_PRIORITY[0] == 'ACCOUNT_SECURITY_ACCESS'
    assert INTENT_PRIORITY[1] == 'DAMAGED_WRONG_MISSING'
    assert INTENT_PRIORITY[-1] == 'FEEDBACK_COMPLAINT_GENERAL'

def test_collision_resolution_security_over_refund():
    # Security must override refund request
    resolved = resolve_intent_collision(['REFUND_RETURN_EXCHANGE', 'ACCOUNT_SECURITY_ACCESS'])
    assert resolved == 'ACCOUNT_SECURITY_ACCESS'

def test_collision_resolution_damaged_over_delivery():
    # Damaged package must take precedence over routine delivery delay
    resolved = resolve_intent_collision(['DELIVERY_STATUS_DELAY', 'DAMAGED_WRONG_MISSING'])
    assert resolved == 'DAMAGED_WRONG_MISSING'

def test_deterministic_risk_stolen_delivered():
    # Delivered but missing must trigger LOST_OR_STOLEN_DELIVERY
    is_risk, cat, reason = evaluate_deterministic_risk('Package shows delivered but never arrived on my porch!')
    assert is_risk is True
    assert cat == 'LOST_OR_STOLEN_DELIVERY'

def test_deterministic_risk_hacked_account():
    is_risk, cat, reason = evaluate_deterministic_risk('Someone hacked my account and changed my password')
    assert is_risk is True
    assert cat == 'ACCOUNT_SECURITY_RISK'

def test_deterministic_risk_routine_inquiry():
    is_risk, cat, reason = evaluate_deterministic_risk('Where is my package? Can you tell me when it will arrive?')
    assert is_risk is False
    assert cat == 'NONE'
