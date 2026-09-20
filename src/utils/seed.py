"""
随机种子设置，保证实验可复现。

用法:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    设置全局随机种子。

    Args:
        seed: 随机种子。
        deterministic: 是否开启完全确定性。
            开启后 cudnn 会变慢，但结果完全可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # transformers 有全局种子设置，装了才设置
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(42)
    print("numpy :", np.random.rand(3))
    print("torch :", torch.rand(3))
    print("random:", [random.random() for _ in range(3)])