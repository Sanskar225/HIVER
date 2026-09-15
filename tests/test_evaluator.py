import pytest
from src.evaluator import compute_ngram_overlap, compute_rouge_l
from src.quality_rubric import DeterministicQualityRubric

def test_ngram_overlap_exact_and_disjoint():
    tokens = ['where', 'is', 'my', 'order']
    assert compute_ngram_overlap(tokens, tokens, 1) == 1.0
    assert compute_ngram_overlap(tokens, ['totally', 'different'], 1) == 0.0

def test_rouge_l_computation():
    ref = ['please', 'visit', 'your', 'orders', 'page']
    hyp = ['please', 'visit', 'your', 'orders', 'page']
    assert compute_rouge_l(hyp, ref) == 1.0
    assert compute_rouge_l(['completely', 'unrelated'], ref) == 0.0

def test_reply_rubric_pass_criteria():
    rubric = DeterministicQualityRubric()
    # High-quality reply with empathy, action, grounded keywords, and safe link
    good_reply = (
        'I am so sorry to hear your package is delayed! You can track real-time courier updates '
        'and your delivery estimate directly from your account at Your Orders: [link]. ^CS'
    )
    res = rubric.evaluate_reply('Where is my package?', good_reply, 'DELIVERY_STATUS_DELAY', 'AUTO_HANDLE')
    assert res['passed'] is True
    assert res['groundedness'] >= 4
    assert res['actionability'] >= 4
    assert res['pii_safety'] == 5

def test_reply_rubric_pii_failure():
    rubric = DeterministicQualityRubric()
    # Reply soliciting sensitive password
    bad_reply = 'Please reply with your credit card details and password so we can assist. ^CS'
    res = rubric.evaluate_reply('I need a refund', bad_reply, 'REFUND_RETURN_EXCHANGE', 'ESCALATE')
    assert res['passed'] is False
    assert res['pii_safety'] == 1
