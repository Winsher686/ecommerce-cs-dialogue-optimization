"""
评估指标。

包含:
    - 文本生成指标：BLEU、ROUGE
    - 语言模型指标：perplexity
    - 业务指标：敏感内容拒答准确率、任务完成率
    - 偏好指标：chosen 胜率

用法:
    from src.evaluation.metrics import compute_all
    report = compute_all(preds, refs)
"""

import math
import re
from collections import Counter
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 文本生成指标
# ------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """中文按字、英文按空白切分，避免客服语料 BLEU/ROUGE 恒为 0。"""
    text = (text or "").strip()
    if not text:
        return []
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(preds: list[str], refs: list[str], n: int = 4) -> float:
    """
    简化版 BLEU，句级平均。
    """
    if not preds:
        return 0.0

    scores = []
    for pred, ref in zip(preds, refs):
        p_tokens = _tokenize(pred)
        r_tokens = _tokenize(ref)
        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue

        precisions = []
        for i in range(1, n + 1):
            p_ng = _ngrams(p_tokens, i)
            r_ng = _ngrams(r_tokens, i)
            overlap = sum((p_ng & r_ng).values())
            total = max(sum(p_ng.values()), 1)
            precisions.append(overlap / total)

        if min(precisions) == 0:
            scores.append(0.0)
            continue

        log_sum = sum(math.log(p) for p in precisions) / n
        bp = min(1.0, math.exp(1 - len(r_tokens) / max(len(p_tokens), 1)))
        scores.append(bp * math.exp(log_sum))

    return round(sum(scores) / len(scores), 4)


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(preds: list[str], refs: list[str]) -> float:
    """ROUGE-L，基于最长公共子序列。"""
    if not preds:
        return 0.0
    scores = []
    for pred, ref in zip(preds, refs):
        p = _tokenize(pred)
        r = _tokenize(ref)
        if not p or not r:
            scores.append(0.0)
            continue
        lcs = _lcs(p, r)
        precision = lcs / len(p)
        recall = lcs / len(r)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
    return round(sum(scores) / len(scores), 4)


def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    return {"rouge_l": compute_rouge_l(preds, refs)}


# ------------------------------------------------------------------
# 语言模型指标
# ------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """
    计算 perplexity。需要传入已加载的模型和 tokenizer。
    """
    import torch

    if not texts:
        return 0.0

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.size(1) - 1
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return 0.0
    avg_loss = total_loss / total_tokens
    return round(math.exp(avg_loss), 4)


# ------------------------------------------------------------------
# 业务指标
# ------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    r"傻", r"垃圾", r"滚", r"骗子", r"投诉你",
    r"不知道", r"不归我们", r"随便",
]

REFUSAL_KEYWORDS = [
    "抱歉", "无法", "不能", "不便", "建议您", "请您",
    "帮您转", "为您核实", "需要核实",
]


def _is_sensitive(text: str) -> bool:
    for p in SENSITIVE_PATTERNS:
        if re.search(p, text):
            return True
    return False


def _is_refusal(text: str) -> bool:
    return any(k in text for k in REFUSAL_KEYWORDS)


def compute_refusal_accuracy(preds: list[str], refs: list[str]) -> float:
    """
    敏感内容拒答准确率：
      - 参考回复是拒答（ref 含拒答关键词），预测也拒答 -> 正确
      - 参考不是拒答，预测也不是敏感内容 -> 正确
    """
    if not preds:
        return 0.0
    correct = 0
    for pred, ref in zip(preds, refs):
        ref_refusal = _is_refusal(ref)
        pred_refusal = _is_refusal(pred) and not _is_sensitive(pred)
        if ref_refusal == pred_refusal:
            correct += 1
    return round(correct / len(preds), 4)


def compute_sensitive_rate(preds: list[str]) -> float:
    """预测中出现敏感内容的比率，越低越好。"""
    if not preds:
        return 0.0
    n = sum(1 for p in preds if _is_sensitive(p))
    return round(n / len(preds), 4)


TASK_KEYWORDS = {
    "logistics": ["物流", "快递", "订单", "发货", "签收"],
    "refund": ["退款", "退货", "售后", "换货"],
    "invoice": ["发票", "开票"],
    "promotion": ["优惠", "活动", "优惠券", "满减"],
    "consult": ["对比", "推荐", "哪个好", "区别"],
}


def _detect_task(text: str) -> Optional[str]:
    for task, kws in TASK_KEYWORDS.items():
        if any(k in text for k in kws):
            return task
    return None


def compute_task_completion_rate(preds: list[str], refs: list[str]) -> float:
    """
    任务完成率：预测回复中是否覆盖了参考回复对应的任务关键词。
    """
    if not preds:
        return 0.0
    correct = 0
    for pred, ref in zip(preds, refs):
        task = _detect_task(ref)
        if task is None:
            correct += 1
            continue
        kws = TASK_KEYWORDS[task]
        if any(k in pred for k in kws):
            correct += 1
    return round(correct / len(preds), 4)


# ------------------------------------------------------------------
# 偏好指标
# ------------------------------------------------------------------

def compute_preference_winrate(
    preds: list[str],
    refs: list[str],
    rejected: list[str],
) -> float:
    """
    简化版偏好胜率：
      用 ROUGE-L 作为代理分数，比较 pred 与 chosen / rejected 的相似度。
      pred 更接近 chosen 则算胜。
    """
    if not preds:
        return 0.0
    wins = 0
    for p, c, r in zip(preds, refs, rejected):
        s_c = compute_rouge_l([p], [c])
        s_r = compute_rouge_l([p], [r])
        if s_c >= s_r:
            wins += 1
    return round(wins / len(preds), 4)


# ------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------

def compute_all(
    preds: list[str],
    refs: list[str],
    rejected: Optional[list[str]] = None,
    model=None,
    tokenizer=None,
) -> dict:
    report = {
        "bleu": compute_bleu(preds, refs),
        "rouge": compute_rouge(preds, refs),
        "refusal_accuracy": compute_refusal_accuracy(preds, refs),
        "sensitive_rate": compute_sensitive_rate(preds),
        "task_completion_rate": compute_task_completion_rate(preds, refs),
    }
    if rejected:
        report["preference_winrate"] = compute_preference_winrate(preds, refs, rejected)
    if model is not None and tokenizer is not None:
        report["perplexity"] = compute_perplexity(model, tokenizer, preds)
    return report