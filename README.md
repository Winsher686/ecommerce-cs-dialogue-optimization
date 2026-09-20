# ecommerce-cs-dialogue-optimization

电商客服场景下的对话模型微调与部署工具集。支持 Qwen3-8B 的 LoRA SFT、DPO 偏好对齐、AWQ 量化、vLLM 推理，以及基于 LoRA 热加载的灰度发布。

## 为什么做这个

电商客服场景对模型有几个特殊要求：

- 多轮对话中要理解模糊、口语化的表达
- 涉及退款、投诉时要有合适的拒答和引导
- 回复不能机械、重复，也不能出现敏感内容
- 推理成本要可控，不能上很重的方案

直接用通用大模型效果不稳定，全量微调成本又太高。这个项目用 LoRA + DPO 的组合，在单卡 A10G 上就能跑推理，训练也只需要几张 A800。

## 特性

- **数据流水线**：清洗、脱敏、审计、自动负例挖掘，支持从原始日志到训练数据的完整流程
- **两阶段训练**：LoRA SFT 打基础，DPO 对齐偏好，SFT 阶段混入通用指令数据缓解遗忘
- **参数高效**：LoRA 微调，只训练少量参数，显存占用可控
- **推理优化**：vLLM + AWQ 4-bit + FlashAttention，单卡即可承载
- **动态灰度**：LoRA 热加载 + Nginx 分流，支持不重启切换模型、按比例放量、快速回滚
- **可观测**：Prometheus 指标暴露，K8s 环境下自动被 ServiceMonitor 采集

## 环境要求

### 数据流水线

- Python 3.10+
- 8GB 内存
- 无需 GPU

### 训练

- 1× 24GB 以上 GPU（SFT，QLoRA 可更低）
- 1× 80GB 或 2× 40GB GPU（DPO）
- 64GB 以上系统内存

### 推理

- 1× 12GB 以上 GPU
- 推荐 A10G / A100

## 安装

```bash
git clone https://github.com/Winsher686/ecommerce-cs-dialogue-optimization.git
cd ecommerce-cs-dialogue-optimization

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

如果只需要跑数据流水线，可以只装轻量依赖：

```bash
pip install -r requirements-mvp.txt
```

## 使用

### 1. 准备数据

项目不附带真实数据。本地可以用生成器造模拟数据跑通流程：

```bash
python -m src.data.preprocess --generate-demo --demo-size 2000
```

数据格式为 JSON Lines，每条记录是一段多轮对话：

```json
{
  "messages": [
    {"role": "user", "content": "我的订单什么时候到？"},
    {"role": "assistant", "content": "请提供订单号，我帮您查询物流进度。"}
  ],
  "source": "demo"
}
```

如果要接入自己的数据，把它放到 `data/raw/`，然后：

```bash
python -m src.data.preprocess --input data/raw/your_logs.jsonl --output data/processed/cleaned.jsonl
```

### 2. 数据审计

```bash
python -m src.data.audit --input data/processed/cleaned.jsonl --output docs/data_card.md
```

会生成一份数据卡，包含轮数分布、长度分布、敏感词命中、质量问题统计。

### 3. 自动负例挖掘

```bash
python -m src.data.mine_negatives --input data/processed/cleaned.jsonl --output data/processed/negative_candidates.jsonl
```

默认走规则筛选（长度、模板、敏感词、重复度）。如果要启用 Embedding 相似度筛选，需要先装 `sentence-transformers`，然后去掉 `--no-embedding`。

### 4. 构建训练数据集

```bash
python -m src.data.build_dataset
```

会生成两个文件：

- `data/processed/sft_dataset.jsonl`：`{prompt, response}`
- `data/processed/dpo_dataset.jsonl`：`{prompt, chosen, rejected}`

### 5. 混入通用指令数据

```bash
python -m src.data.mix_general_data --generate-demo --demo-size 300
```

输出 `data/processed/sft_dataset_mixed.jsonl`，通用数据占比 10%。

### 6. SFT 训练

```bash
bash scripts/run_sft.sh
```

配置文件在 `configs/sft.yaml`，关键参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `model_name_or_path` | 底座模型 | `Qwen/Qwen3-8B` |
| `learning_rate` | 学习率 | `2e-5` |
| `lr_scheduler_type` | 调度器 | `cosine` |
| `warmup_ratio` | 预热比例 | `0.05` |
| `lora.r` | LoRA 秩 | `8` |
| `lora.alpha` | LoRA alpha | `16` |

训练时只对 assistant 回复部分计算损失，prompt 部分的 label 会被置为 `-100`。

### 7. DPO 训练

```bash
bash scripts/run_dpo.sh
```

配置文件在 `configs/dpo.yaml`，关键参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `model_name_or_path` | SFT 后的 LoRA 路径 | `outputs/sft` |
| `beta` | KL 惩罚强度 | `0.1` |
| `loss_type` | loss 类型，支持 `dpo` / `ipo` | `dpo` |
| `learning_rate` | 学习率 | `1e-5` |

DPO 会同时加载策略模型和参考模型，参考模型参数冻结。

### 8. 量化

```bash
bash scripts/run_quantize.sh
```

用 AWQ 把模型量化到 4-bit，输出到 `outputs/awq/`。

### 9. 启动推理服务

```bash
bash scripts/run_api.sh
```

默认监听 8000 端口，Swagger 文档在 `http://localhost:8000/docs`。

如果本地没有 GPU，或者只是想验证接口，可以用 mock 引擎：

```bash
export USE_MOCK_ENGINE=1
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
bash scripts/run_api.sh
```

### 10. 评估

```bash
bash scripts/run_eval.sh
```

会在测试集上生成回复并计算指标，输出到 `outputs/eval/report.json`。

支持的指标：

- BLEU、ROUGE-L
- 拒答准确率
- 敏感内容命中率
- 任务完成率
- 偏好胜率

## 配置

所有配置都在 `configs/` 下：

- `sft.yaml`：SFT 训练
- `dpo.yaml`：DPO 训练
- `inference.yaml`：推理与评估
- `mvp/`：小模型 MVP 配置，用于本地快速验证

配置支持环境变量覆盖。比如：

```bash
MODEL_NAME_OR_PATH=outputs/sft bash scripts/run_dpo.sh
```

## 部署

### Docker Compose

```bash
cd docker
docker compose up -d
```

会启动三个服务：

- `api-base`：基座模型服务，8001
- `api-finetuned`：微调模型服务，8002
- `nginx`：灰度入口，80

Nginx 的分流比例在 `docker/nginx.conf` 的 `split_clients` 里配置。

### Kubernetes

```bash
kubectl apply -f k8s/
```

包含：

- `deployment.yaml`：2 副本，带健康检查
- `service.yaml`：ClusterIP
- `ingress.yaml`：对外暴露
- `configmap.yaml`：非敏感配置
- `pvc.yaml`：模型存储卷
- `servicemonitor.yaml`：Prometheus 采集

## 灰度与回滚

灰度有两层控制：

**Nginx 层**：修改 `split_clients` 里的比例，重新加载配置。

**应用层**：通过接口动态调整，不需要重启。

```bash
# 查看当前状态
curl http://localhost:8000/router/status

# 调整微调模型流量比例
curl -X POST "http://localhost:8000/router/ratio?ratio=0.3"

# 快速回滚，100% 走基座
curl -X POST http://localhost:8000/router/rollback

# 恢复灰度
curl -X POST http://localhost:8000/router/recover
```

LoRA 适配器管理：

```bash
# 查看已注册的适配器
curl http://localhost:8000/lora/list
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 发送消息 |
| POST | `/reset` | 清空会话 |
| GET | `/router/status` | 查看灰度状态 |
| POST | `/router/ratio` | 调整灰度比例 |
| POST | `/router/rollback` | 回滚 |
| POST | `/router/recover` | 恢复 |
| GET | `/lora/list` | 列出适配器 |
| GET | `/metrics` | Prometheus 指标 |

`/chat` 请求示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我要退货", "session_id": "u1"}'
```

响应：

```json
{
  "reply": "退货可以在订单页申请，选择原因后会有快递上门取件。",
  "target": "finetuned",
  "session_id": "u1"
}
```

## 项目结构

```
.
├── configs/              # 配置文件
├── data/                 # 数据目录，真实数据不提交
├── docker/               # 容器化与 Nginx
├── docs/                 # 文档
├── k8s/                  # Kubernetes 部署
├── scripts/              # 一键脚本
├── src/
│   ├── data/             # 数据处理
│   ├── training/         # 训练
│   ├── evaluation/       # 评估
│   ├── inference/        # 推理与服务
│   └── utils/            # 工具
├── tests/                # 测试
├── Makefile
└── requirements.txt
```

## 常见问题

**Q: 数据从哪里来？**

项目不附带真实数据。本地可以用 `--generate-demo` 生成模拟数据跑通流程。接入自己的数据时，请确保已做脱敏。

**Q: 没有 GPU 能跑吗？**

数据流水线完全不需要 GPU。训练和推理需要 GPU，但可以用小模型验证逻辑，配置在 `configs/mvp/` 下。

**Q: 训练需要多长时间？**

参考配置下，SFT 约 2 小时（1× A800），DPO 约 3 小时（2× A800）。实际时间取决于数据量和硬件。

**Q: 为什么 SFT 和 DPO 都要只对回复部分算 loss？**

用户输入是条件，不是要拟合的目标。把它算进 loss 会让模型学习去预测用户的下一句话，这不是我们想要的。

**Q: beta 参数怎么调？**

beta 控制 KL 惩罚强度。值越大更新越保守，越小越激进。一般从 0.1 开始试。

**Q: 推理时怎么切换模型？**

通过 `GrayRouter` 按比例分流，或者调 `/router/ratio` 动态调整。LoRA 适配器可以通过 `LoRAManager` 热加载，不需要重启。

## 开发

```bash
# 跑测试
pytest tests/ -v

# 语法检查
python -m compileall src

# 代码格式（可选）
ruff check src
```

CI 配置在 `.github/workflows/ci.yml`，每次 push 会自动跑语法检查和单元测试。

## 已知限制

- 自动负例挖掘的 Embedding 部分依赖 `sentence-transformers`，未安装时自动跳过
- AWQ 量化需要 `autoawq`，Windows 下可能装不上，建议在 Linux 环境执行
- vLLM 同理，生产部署建议用 Linux + NVIDIA GPU

## 参考

- [Qwen3](https://github.com/QwenLM/Qwen3)
- [TRL](https://github.com/huggingface/trl)
- [vLLM](https://github.com/vllm-project/vllm)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

## License

MIT
