# 电商智能客服闲聊项目（DPO）

基于 Qwen3-8B 的电商客服多轮对话优化项目，采用 SFT + DPO 微调，让模型在任务型对话、知识库问答、闲聊修复场景中输出更符合客户偏好的回复。

## 技术栈

- 底座模型：Qwen3-8B
- 微调方法：SFT + DPO
- 训练框架：Hugging Face Transformers / TRL / PEFT
- 部署：FastAPI + Docker + Kubernetes

## 目录结构

- configs/：SFT、DPO、推理配置
- data/：原始数据与处理后数据，不提交真实数据
- src/：数据、训练、评估、推理源码
- scripts/：训练和部署脚本
- docker/：Dockerfile 与 compose
- k8s/：Kubernetes 部署配置
- docs/：架构、数据卡、实验记录
- outputs/：训练输出，不提交模型权重

## 快速开始

1. 安装依赖：pip install -r requirements.txt
2. 准备数据：放入 data/raw/，处理后输出到 data/processed/
3. SFT 训练：bash scripts/run_sft.sh
4. DPO 训练：bash scripts/run_dpo.sh
5. 启动 API：bash scripts/run_api.sh

## 项目亮点

- SFT 阶段：只对客服回答部分计算损失，约 50 万轮内部多轮客服对话
- DPO 阶段：基于正负例偏好数据约 10 万对，beta=0.1，2×A800 约 3 小时
- 工程实现：基于 Hugging Face TRL 的 DPOTrainer，支持重写 DPO loss
