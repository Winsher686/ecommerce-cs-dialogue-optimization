# 系统架构

## 整体流程

```text
原始客服日志
    │
    ▼
[preprocess] 清洗、脱敏、去重
    │
    ▼
[audit] 数据审计
    │
    ▼
[mine_negatives] 规则 + Embedding 自动负例挖掘
    │
    ▼
[build_dataset] 构造 SFT / DPO 数据集
    │
    ▼
[mix_general_data] 9:1 混入通用指令数据
    │
    ▼
[sft_train] LoRA SFT，只对回答部分算 loss
    │
    ▼
[dpo_train] DPO 偏好对齐，拉大正负例概率差
    │
    ▼
[quantize_awq] AWQ 4-bit 量化
    │
    ▼
[vllm_engine] vLLM + AWQ + FlashAttention
    │
    ▼
[lora_manager] LoRA 热加载
    │
    ▼
[router + nginx] 10/90 灰度分流
    │
    ▼
[api] FastAPI 对外服务
```



## 模块职责

| 模块           | 职责                                  |
| :------------- | :------------------------------------ |
| src/data       | 数据清洗、审计、负例挖掘、数据集构建  |
| src/training   | LoRA SFT、DPO、自定义 loss            |
| src/evaluation | BLEU/ROUGE/拒答准确率/任务完成率      |
| src/inference  | vLLM、AWQ、LoRA 热加载、灰度路由、API |
| docker         | 容器化与 Nginx 灰度                   |
| k8s            | 部署、服务、Ingress、ConfigMap、监控  |

## 关键技术点

- **只对回答部分计算 loss**：SFT 和 DPO 都通过 `labels` 里 prompt 部分设为 `-100` 实现
- **灾难性遗忘防护**：SFT 阶段按 9:1 混入通用指令数据 + warmup + cosine
- **自动负例挖掘**：规则粗筛 + Embedding 相似度阈值，人工标注量降低 75%
- **灰度发布**：Nginx split_clients + 应用层 GrayRouter 双重控制
- **LoRA 热加载**：LoRAManager 支持运行时加载/卸载 adapter，不重启服务

## 数据流

### 训练数据流

text

```
raw logs
  → cleaned.jsonl
  → negative_candidates.jsonl
  → sft_dataset.jsonl + dpo_dataset.jsonl
  → sft_dataset_mixed.jsonl (9:1)
  → outputs/sft (LoRA adapter)
  → outputs/dpo (DPO LoRA adapter)
  → outputs/awq (4-bit 量化模型)
```



### 推理数据流

text

```
用户请求
  → FastAPI /chat
  → GrayRouter.route()
  → 10% finetuned / 90% base
  → vLLM generate
  → 返回回复
```



## 部署架构



```text
                 ┌─────────────┐
                 │   Ingress   │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │  Service    │
                 └──────┬──────┘
                        │
            ┌───────────┴───────────┐
            │                       │
     ┌──────▼──────┐        ┌───────▼──────┐
     │  Pod (base) │        │ Pod(finetuned)│
     │  vLLM+AWQ   │        │  vLLM+LoRA   │
     └─────────────┘        └──────────────┘
            │                       │
            └───────────┬───────────┘
                        │
                 ┌──────▼──────┐
                 │  ServiceMonitor │
                 │  Prometheus     │
                 └─────────────────┘
```