"""评估指标单元测试。"""

from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge_l,
    compute_refusal_accuracy,
    compute_sensitive_rate,
    compute_task_completion_rate,
    compute_preference_winrate,
    compute_all,
)


def test_bleu_identical():
    preds = ["你好世界"]
    refs = ["你好世界"]
    assert compute_bleu(preds, refs) > 0.9


def test_bleu_completely_different():
    preds = ["abc"]
    refs = ["xyz"]
    assert compute_bleu(preds, refs) == 0.0


def test_rouge_l_identical():
    preds = ["请提供订单号"]
    refs = ["请提供订单号"]
    assert compute_rouge_l(preds, refs) == 1.0


def test_rouge_l_partial():
    preds = ["请提供订单号我帮您查询"]
    refs = ["请提供订单号"]
    score = compute_rouge_l(preds, refs)
    assert 0 < score < 1


def test_refusal_accuracy():
    preds = ["抱歉，我无法处理", "请提供订单号"]
    refs = ["抱歉，请您稍等", "请提供订单号"]
    score = compute_refusal_accuracy(preds, refs)
    assert score == 1.0


def test_sensitive_rate():
    preds = ["你好", "不知道。", "请提供订单号"]
    score = compute_sensitive_rate(preds)
    assert 0 < score < 1


def test_task_completion_rate():
    preds = ["请提供订单号，我帮您查询物流"]
    refs = ["请提供订单号查询物流"]
    score = compute_task_completion_rate(preds, refs)
    assert score == 1.0


def test_preference_winrate():
    preds = ["请提供订单号，我帮您查询物流"]
    refs = ["请提供订单号，我帮您查询物流"]
    rejected = ["不知道。"]
    score = compute_preference_winrate(preds, refs, rejected)
    assert score == 1.0


def test_compute_all():
    preds = ["请提供订单号"]
    refs = ["请提供订单号"]
    report = compute_all(preds, refs)
    assert "bleu" in report
    assert "rouge" in report
    assert "refusal_accuracy" in report
    assert "task_completion_rate" in report