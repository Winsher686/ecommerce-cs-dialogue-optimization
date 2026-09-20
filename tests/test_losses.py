"""DPO loss 单测。torch 不可用时自动跳过。"""

import pytest

torch = pytest.importorskip("torch")

from src.training.losses import compute_logps, dpo_loss, get_answer_mask


def test_dpo_loss_prefers_chosen():
    policy_c = torch.tensor([ -1.0, -1.0])
    policy_r = torch.tensor([ -3.0, -3.0])
    ref_c = torch.tensor([ -1.5, -1.5])
    ref_r = torch.tensor([ -1.5, -1.5])
    loss, cr, rr = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    assert loss.ndim == 0
    assert (cr > rr).all()


def test_answer_mask_and_logps():
    logits = torch.zeros(1, 4, 5)
    logits[:, :, 1] = 10.0
    labels = torch.tensor([[-100, 1, 1, 1]])
    mask = get_answer_mask(labels)
    assert mask.sum().item() == 3
    logps = compute_logps(logits, labels)
    assert logps.shape == (1,)
