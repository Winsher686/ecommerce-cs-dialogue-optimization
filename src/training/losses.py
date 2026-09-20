"""
自定义 DPO loss。

支持:
    - 标准 DPO
    - IPO 变体
    - 只对回答部分 token 计算 logps

用法:
    from src.training.losses import dpo_loss, compute_logps, get_answer_mask
"""

from typing import Optional

import torch
import torch.nn.functional as F


def get_answer_mask(
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    根据 labels 生成回答部分的 mask。
    labels 中 -100 的位置是 prompt，其余是回答。

    Returns:
        mask: 与 labels 同形状，回答位置为 1，其他为 0
    """
    return (labels != ignore_index).float()


def compute_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    计算序列中回答部分 token 的对数概率之和。

    Args:
        logits: [B, L, V]
        labels: [B, L]
        ignore_index: prompt 位置的 label 值

    Returns:
        logps: [B] 每条样本回答部分的 logp 之和
    """
    # shift: 预测下一个 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 取每个位置对应 label 的 logp
    gathered = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)

    mask = (shift_labels != ignore_index).float()
    gathered = gathered * mask

    return gathered.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "dpo",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 DPO loss。

    Args:
        policy_chosen_logps:   策略模型对正例的 logps   [B]
        policy_rejected_logps: 策略模型对负例的 logps   [B]
        ref_chosen_logps:      参考模型对正例的 logps   [B]
        ref_rejected_logps:    参考模型对负例的 logps   [B]
        beta:                  KL 惩罚强度
        loss_type:             dpo / ipo

    Returns:
        loss, chosen_rewards, rejected_rewards
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    if loss_type == "dpo":
        loss = -F.logsigmoid(beta * logits)
    elif loss_type == "ipo":
        # IPO: 直接回归到目标 margin
        loss = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return loss.mean(), chosen_rewards, rejected_rewards