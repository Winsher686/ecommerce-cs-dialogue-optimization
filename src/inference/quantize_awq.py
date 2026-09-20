"""
AWQ 4-bit 量化。

用法:
    python -m src.inference.quantize_awq \
        --model  outputs/dpo \
        --output outputs/awq \
        --bits   4
"""

import argparse
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    calib_samples: int = 128,
) -> None:
    """
    用 autoawq 对模型做 4-bit 量化。

    Args:
        model_path:    原始模型或 LoRA 合并后的模型路径
        output_path:   量化后模型输出路径
        bits:          量化位宽，通常 4
        group_size:    分组大小
        zero_point:    是否使用 zero point
        calib_samples: 校准样本数
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("请先安装 autoawq: pip install autoawq")
        raise

    logger.info(f"loading model: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "w_bit": bits,
        "q_group_size": group_size,
        "zero_point": zero_point,
        "version": "GEMM",
    }

    logger.info(f"quantizing to {bits}-bit AWQ")
    model.quantize(tokenizer, quant_config=quant_config)

    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"AWQ model saved to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="outputs/dpo")
    p.add_argument("--output", type=str, default="outputs/awq")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize_awq(args.model, args.output, args.bits, args.group_size)