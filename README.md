<div align="center">

# 🔥 LLM Interview Hot 100

### *LLM 时代的 Hot 100*

**大模型面试手撕代码 · 社区投票驱动 · 每日更新排行**

[![GitHub stars](https://img.shields.io/github/stars/your-username/llm-interview-hot-100?style=social)](https://github.com/your-username/llm-interview-hot-100)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Vote](https://img.shields.io/badge/🗳️-参与投票-ff6b6b)](https://your-username.github.io/llm-interview-hot-100)

<p align="center">
  <strong>📖 背完这 100 题，大模型面试手撕不再慌</strong>
</p>

---

**"面试官让你手撕 Multi-Head Attention，你还在紧张？"**

**"PPO、DPO、GRPO 的区别，你能写出来吗？"**

**"KV Cache 怎么实现？Flash Attention 的核心思想是什么？"**

**👉 [在线投票：哪道题最常被考？](https://your-username.github.io/llm-interview-hot-100) 👈**

</div>

---

## ✨ 项目特点

| 特点 | 描述 |
|:---:|:---|
| 🎯 **面试真题** | 社区投票驱动，真实反映面试频率 |
| 📝 **详细注释** | 每行代码都有清晰注释，理解原理而非死记 |
| 🔥 **工业级代码** | 包含数值稳定性、边界处理等生产细节 |
| 🆚 **方法对比** | 同类方法一览对比，记住一个就记住一组 |
| ❓ **QA 答疑** | 预判你的困惑，看完豁然开朗 |

---

## 📋 完整题目清单

> **图例：** 🔥🔥🔥 必考 ｜ 🔥🔥 高频 ｜ 🔥 偶尔 ｜ 无标记 了解即可
>
> **👉 [参与投票](https://your-username.github.io/llm-interview-hot-100)**，用你的面试经历帮助社区校准 Hot 程度！

### 📖 LLM 基础 → [查看](docs/00-llm-basics.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 1 | 梯度与反向传播 | 🔥🔥 | ⭐⭐ | 链式法则手推，深度学习根基 |
| 2 | 线性回归 | 🔥 | ⭐ | `y = Wx + b`，最简单的模型 |
| 3 | 逻辑回归 | 🔥🔥 | ⭐⭐ | `sigmoid(Wx + b)`，二分类基础 |
| 4 | Softmax 回归 | 🔥 | ⭐⭐ | 多分类，LLM 输出层 |
| 5 | MLP 多层感知机 | 🔥🔥 | ⭐⭐ | 万能近似器，FFN 的基础 |
| 6 | 激活函数 | 🔥🔥 | ⭐ | ReLU/GELU/SiLU 及其梯度 |

### 🧠 Attention 机制 → [查看](docs/01-attention.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 7 | Scaled Dot-Product Attention | 🔥🔥🔥 | ⭐⭐⭐ | `softmax(QK^T/√d)V`，一切的基础 |
| 8 | Multi-Head Attention | 🔥🔥🔥 | ⭐⭐⭐⭐ | 多头并行，不同子空间的注意力 |
| 9 | Causal Mask | 🔥🔥🔥 | ⭐⭐ | 下三角掩码，防止看到未来 |
| 10 | Grouped Query Attention (GQA) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Q头数 > KV头数，LLaMA2 主流 |
| 11 | Multi-Query Attention (MQA) | 🔥🔥 | ⭐⭐⭐ | 所有 Q 共享一组 KV |
| 12 | Flash Attention | 🔥🔥 | ⭐⭐⭐⭐⭐ | 分块计算，IO 感知，内存 O(N) |
| 13 | KV Cache | 🔥🔥🔥 | ⭐⭐⭐⭐ | 缓存历史 KV，避免重复计算 |
| 14 | Cross Attention | 🔥 | ⭐⭐⭐ | Q 来自 decoder，KV 来自 encoder |

### 📏 归一化层 → [查看](docs/02-normalization.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 15 | Layer Normalization | 🔥🔥🔥 | ⭐⭐ | 沿特征维度归一化，Transformer 标配 |
| 16 | RMS Normalization | 🔥🔥🔥 | ⭐⭐ | 去掉均值只除 RMS，LLaMA 用 |
| 17 | Batch Normalization | 🔥 | ⭐⭐ | 沿 batch 维度归一化，CNN 常用 |
| 18 | Pre-Norm vs Post-Norm | 🔥🔥 | ⭐ | Pre-Norm 训练稳定，现代 LLM 主流 |

### 📍 位置编码 → [查看](docs/03-position-encoding.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 19 | Sinusoidal PE | 🔥 | ⭐⭐ | sin/cos 固定编码，原始 Transformer |
| 20 | Learnable PE | 🔥 | ⭐ | 可学习的嵌入，BERT/GPT |
| 21 | RoPE 旋转位置编码 | 🔥🔥🔥 | ⭐⭐⭐⭐ | 复数旋转，相对位置，LLM 主流 |
| 22 | ALiBi | 🔥🔥 | ⭐⭐⭐ | attention 加线性偏置，长度外推好 |

### 🎲 采样策略 → [查看](docs/04-sampling.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 23 | Greedy Decoding | 🔥 | ⭐ | 每步选 argmax，确定性输出 |
| 24 | Temperature Sampling | 🔥🔥🔥 | ⭐⭐ | `logits/T` 控制随机性 |
| 25 | Top-k Sampling | 🔥🔥 | ⭐⭐ | 只从 top-k 中采样 |
| 26 | Top-p (Nucleus) Sampling | 🔥🔥🔥 | ⭐⭐⭐ | 累积概率达到 p 后截断 |
| 27 | Beam Search | 🔥🔥 | ⭐⭐⭐ | 保留 k 个最优序列 |

### 📉 损失函数 → [查看](docs/05-loss-functions.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 28 | Cross Entropy Loss | 🔥🔥🔥 | ⭐⭐⭐ | `-log(p_true)`，分类标配 |
| 29 | Language Model Loss | 🔥🔥🔥 | ⭐⭐ | CE 的自回归版，next token prediction |
| 30 | KL Divergence | 🔥🔥 | ⭐⭐⭐ | 分布差异度量，蒸馏/RLHF 用 |
| 31 | MSE Loss | 🔥 | ⭐ | `(y-ŷ)²`，回归任务 |
| 32 | Focal Loss | 🔥 | ⭐⭐⭐ | 降低易分类样本权重 |
| 33 | SFT Loss | 🔥🔥 | ⭐⭐ | 带 mask 的 CE，只算 response |
| 34 | Reward Model Loss | 🔥🔥 | ⭐⭐⭐ | `-log σ(r_w - r_l)`，偏好学习 |
| 35 | Contrastive Loss | 🔥 | ⭐⭐⭐ | 正样本近，负样本远 |

### ⚡ 优化器 → [查看](docs/06-optimizers.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 36 | SGD | 🔥 | ⭐ | 最基础 `w -= lr * grad` |
| 37 | SGD + Momentum | 🔥 | ⭐⭐ | 加动量，加速收敛 |
| 38 | Adam | 🔥🔥🔥 | ⭐⭐⭐ | 自适应学习率，一阶+二阶矩 |
| 39 | AdamW | 🔥🔥🔥 | ⭐⭐⭐ | 解耦权重衰减，LLM 标配 |
| 40 | 学习率调度 | 🔥🔥 | ⭐⭐ | Warmup + Cosine/Linear decay |

### 🎮 强化学习 (RLHF) → [查看](docs/07-reinforcement-learning.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 41 | REINFORCE | 🔥 | ⭐⭐⭐ | 策略梯度基础 `∇log π × R` |
| 42 | GAE | 🔥🔥🔥 | ⭐⭐⭐⭐ | 优势估计，平衡偏差方差 |
| 43 | PPO | 🔥🔥🔥 | ⭐⭐⭐⭐⭐ | clip 限制更新幅度，RLHF 核心 |
| 44 | PPO-Clip | 🔥🔥🔥 | ⭐⭐⭐⭐ | ratio clip 版本 |
| 45 | DPO | 🔥🔥🔥 | ⭐⭐⭐⭐ | 直接偏好优化，无需 RM |
| 46 | GRPO | 🔥🔥🔥 | ⭐⭐⭐⭐⭐ | 组相对策略优化，DeepSeek 用 |
| 47 | KL 惩罚 | 🔥🔥 | ⭐⭐ | 防止偏离参考策略太远 |
| 48 | Reward Shaping | 🔥 | ⭐⭐⭐ | 奖励工程，稀疏 → 稠密 |

### 🚀 高效训练 → [查看](docs/08-efficient-training.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 49 | LoRA | 🔥🔥🔥 | ⭐⭐⭐⭐ | 低秩分解 `W + BA` |
| 50 | QLoRA | 🔥🔥 | ⭐⭐⭐⭐ | LoRA + 4bit 量化 |
| 51 | Gradient Checkpointing | 🔥🔥 | ⭐⭐⭐ | 时间换空间，重计算激活值 |
| 52 | Mixed Precision (FP16/BF16) | 🔥🔥 | ⭐⭐⭐ | 降低显存，加速计算 |
| 53 | Gradient Accumulation | 🔥🔥 | ⭐⭐ | 小 batch 模拟大 batch |

### ⚡ 推理优化 → [查看](docs/09-inference-optimization.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 54 | KV Cache | 🔥🔥🔥 | ⭐⭐⭐⭐ | 缓存历史 KV，加速自回归 |
| 55 | Paged Attention | 🔥🔥 | ⭐⭐⭐⭐ | 分页管理 KV，vLLM 核心 |
| 56 | Speculative Decoding | 🔥🔥 | ⭐⭐⭐⭐ | 小模型猜测 + 大模型验证 |
| 57 | Continuous Batching | 🔥🔥 | ⭐⭐⭐ | 动态 batch，提高吞吐 |
| 58 | Quantization (INT8/INT4) | 🔥🔥 | ⭐⭐⭐ | 量化推理，显存减半+ |

### 🏗️ Transformer 架构 → [查看](docs/10-transformer-architecture.md)

| # | 题目 | Hot | 难度 | 一句话 |
|:---:|:---|:---:|:---:|:---|
| 59 | Encoder-Only (BERT) | 🔥 | ⭐⭐⭐ | 双向注意力，理解任务 |
| 60 | Decoder-Only (GPT) | 🔥🔥🔥 | ⭐⭐⭐ | 因果注意力，生成任务，LLM 主流 |
| 61 | Encoder-Decoder (T5) | 🔥 | ⭐⭐⭐ | 序列到序列，翻译/摘要 |
| 62 | FFN | 🔥🔥 | ⭐⭐ | 两层 MLP，中间扩展 4x |
| 63 | SwiGLU | 🔥🔥 | ⭐⭐⭐ | 门控 FFN，LLaMA 用 |

---

## 🔥 高频 Top 20

> 由社区投票实时产生，每日 00:00 自动更新
>
> **最后更新**: 2025-03-26 (初始排名，等待社区投票)

| 排名 | 题目 | 分类 | 票数 |
|:---:|:---|:---|:---:|
| 🥇 | Multi-Head Attention | Attention | 🔥 — |
| 🥈 | KV Cache | 推理优化 | 🔥 — |
| 🥉 | RoPE 位置编码 | 位置编码 | 🔥 — |
| 4 | PPO | 强化学习 | 🔥 — |
| 5 | DPO | 强化学习 | 🔥 — |
| 6 | GRPO | 强化学习 | 🔥 — |
| 7 | Cross Entropy Loss | 损失函数 | 🔥 — |
| 8 | LayerNorm / RMSNorm | 归一化 | 🔥 — |
| 9 | Adam / AdamW | 优化器 | 🔥 — |
| 10 | Top-p Sampling | 采样策略 | 🔥 — |
| 11 | LoRA | 高效训练 | 🔥 — |
| 12 | GQA / MQA | Attention | 🔥 — |
| 13 | Causal Mask | Attention | 🔥 — |
| 14 | GAE | 强化学习 | 🔥 — |
| 15 | Beam Search | 采样策略 | 🔥 — |
| 16 | Flash Attention | Attention | 🔥 — |
| 17 | Decoder-Only 架构 | 架构 | 🔥 — |
| 18 | Gradient Checkpointing | 高效训练 | 🔥 — |
| 19 | Speculative Decoding | 推理优化 | 🔥 — |
| 20 | SwiGLU | 架构 | 🔥 — |

---

## 📖 学习路线

### ⚡ 面试前突击（1-2 天）

```
高频 Top 20 → Attention 全部 → 损失函数 → 采样策略
```

### 📚 系统学习（1 周）

```
按分类顺序学习，每个分类先看原理再看代码
```

### 🎯 按方向准备

| 方向 | 重点章节 |
|:---|:---|
| **大厂基础岗** | Attention + 损失函数 + 优化器 + 归一化 |
| **RLHF 方向** | 强化学习全部 + 损失函数 |
| **推理优化** | 推理优化 + Attention + 高效训练 |
| **训练优化** | 高效训练 + 优化器 + 架构 |

---

## 🗳️ 参与投票

**你的面试经历很重要！** 帮助社区校准题目的真实热度。

**👉 [前往投票页面](https://your-username.github.io/llm-interview-hot-100) 👈**

- 🗳️ 投票给你面试中遇到过的题目
- 🏆 实时排行榜，社区真实数据
- 💬 分享你的面试经历和建议

---

## 🤝 贡献指南

欢迎贡献新题目、修复错误、改进文档！

1. **Fork** 本仓库
2. 创建特性分支 `git checkout -b feature/new-topic`
3. 提交更改 `git commit -m 'Add: XXX'`
4. 推送分支 `git push origin feature/new-topic`
5. 提交 **Pull Request**

### 题目格式

```markdown
## 题目名称

### 🎯 核心思想
一句话说明

### 📝 实现代码
带详细注释的代码

### 🔍 复杂度分析
时间/空间复杂度

### 💡 面试追问
常见追问和答案
```

---

## 📜 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)

---

## ⭐ Star History

如果这个项目对你有帮助，请给一个 Star ⭐ 支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/llm-interview-hot-100&type=Date)](https://star-history.com/#your-username/llm-interview-hot-100&Date)

---

<div align="center">

**Made with ❤️ for LLM Interview Preparation**

*LLM 时代的 Hot 100*

**#LLMHot100**

[Report Bug](https://github.com/your-username/llm-interview-hot-100/issues) · [Request Feature](https://github.com/your-username/llm-interview-hot-100/issues) · [参与投票](https://your-username.github.io/llm-interview-hot-100)

</div>
