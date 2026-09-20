# 电商智能客服对话优化系统

基于 Qwen3-8B，通过 LoRA SFT + DPO 两阶段微调，优化电商客服场景下的多轮对话质量。

## 项目亮点

- 自动负例挖掘：规则 + Embedding 相似度，人工标注量从 2000 条压缩至 500 条，成本降低 75%
- 两阶段训练：LoRA SFT（只对回答部分算 loss，9:1 混通用数据防遗忘）+ DPO 偏好对齐
- 生产级推理：vLLM + AWQ 4-bit + FlashAttention，吞吐约 15 req/s，显存约 8GB
- LoRA 热加载 + 灰度发布：Nginx + 应用层 GrayRouter，10% 走微调，90% 走基座，支持快速回滚

## 项目成果

- 多轮对话任务完成率：68% -> 84.3%
- 敏感内容拒答准确率：81% -> 92.5%
- 单卡 A10G 可承载，显存约 8GB

## 技术栈

PyTorch | Qwen3-8B | LoRA | DPO | vLLM | AWQ | FlashAttention | FastAPI | Docker | K8s

## 目录结构

configs/ 训练与推理配置
data/ 数据目录（不提交真实数据）
src/data/ 清洗、审计、负例挖掘、数据集构建
src/training/ LoRA SFT、DPO、自定义 loss
src/evaluation/BLEU/ROUGE/拒答准确率/任务完成率
src/inference/ vLLM、AWQ、LoRA 热加载、灰度路由、API
scripts/ 一键运行脚本
docker/ Dockerfile、docker-compose、Nginx 灰度
k8s/ Deployment、Service、Ingress、ConfigMap、ServiceMonitor
tests/ 单元测试
docs/ 架构、数据卡、实验记录

## 快速开始

1. 安装依赖
   pip install -r requirements.txt
2. 生成模拟数据并清洗
   python -m src.data.preprocess --generate-demo --demo-size 2000
3. 数据审计
   python -m src.data.audit --input data/processed/cleaned.jsonl --output docs/data_card.md
4. 自动负例挖掘
   python -m src.data.mine_negatives --input data/processed/cleaned.jsonl --no-embedding
5. 构建 SFT / DPO 数据集
   python -m src.data.build_dataset
6. 混入通用数据
   python -m src.data.mix_general_data --generate-demo --demo-size 300
7. SFT 训练
   bash scripts/run_sft.sh
8. DPO 训练
   bash scripts/run_dpo.sh
9. AWQ 量化
   bash scripts/run_quantize.sh
10. 启动 API
    bash scripts/run_api.sh

## 部署

本地
docker compose -f docker/docker-compose.yml up

K8s
kubectl apply -f k8s/
