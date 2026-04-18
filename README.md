<div align="center">

# 🔥 LLM Interview Hot 100

### *The Hot 100 for the LLM Era*

**Hand-torn Code for LLM Interviews · Community-Driven Voting · Real-time Rankings**

[![GitHub stars](https://img.shields.io/github/stars/cdhx/LLM-Code-Hot-100?style=social)](https://github.com/cdhx/LLM-Code-Hot-100)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Vote](https://img.shields.io/badge/🗳️-Vote_Now-ff6b6b)](https://cdhx.github.io/LLM-Code-Hot-100)

**[English](README.md) | [中文](README_zh.md)**

<p align="center">
  <strong>📖 Master these 100 topics, ace your LLM interview coding challenges</strong>
</p>

---

**"Asked to implement Multi-Head Attention from scratch?"**

**"Can you write PPO, DPO, GRPO and explain the differences?"**

**"How does KV Cache work? What's the core idea of Flash Attention?"**

**👉 [Vote: Which topics appear most often?](https://cdhx.github.io/LLM-Code-Hot-100) 👈**

</div>

---

## ✨ Features

| Feature | Description |
|:---:|:---|
| 🎯 **Real Interview Questions** | Rankings driven by community votes |
| 📝 **Detailed Comments** | Every line of code clearly annotated |
| 🔥 **Production-Ready** | Numerical stability, edge cases handled |
| 🆚 **Method Comparisons** | Side-by-side comparisons of similar methods |
| ❓ **Q&A Sections** | Common questions answered proactively |

---

## 📋 Complete Topic List

> **Legend:** 🔥🔥🔥 Must Know ｜ 🔥🔥 High Frequency ｜ 🔥 Occasional ｜ No mark = Good to know
>
> **👉 [Vote Now](https://cdhx.github.io/LLM-Code-Hot-100)** to help calibrate the real interview frequency!

### 📖 LLM Basics → [View](docs/00-llm-basics.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 1 | [Gradient & Backprop](docs/00-llm-basics.md#梯度与反向传播) | 🔥🔥 | ⭐⭐ | Chain rule, foundation of deep learning |
| 2 | [Linear Regression](docs/00-llm-basics.md#线性回归) | 🔥 | ⭐ | `y = Wx + b`, simplest model |
| 3 | [Logistic Regression](docs/00-llm-basics.md#逻辑回归) | 🔥🔥 | ⭐⭐ | `sigmoid(Wx + b)`, binary classification |
| 4 | [Softmax Regression](docs/00-llm-basics.md#回归-vs-分类) | 🔥 | ⭐⭐ | Multi-class, LLM output layer |
| 5 | [MLP](docs/00-llm-basics.md#mlp-多层感知机) | 🔥🔥 | ⭐⭐ | Universal approximator, FFN basis |
| 6 | [Activation Functions](docs/00-llm-basics.md#激活函数) | 🔥🔥 | ⭐ | ReLU/GELU/SiLU and gradients |

### 🧠 Attention Mechanisms → [View](docs/01-attention.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 7 | [Scaled Dot-Product Attention](docs/01-attention.md#scaled-dot-product-attention) | 🔥🔥🔥 | ⭐⭐⭐ | `softmax(QK^T/√d)V`, the foundation |
| 8 | [Multi-Head Attention](docs/01-attention.md#multi-head-attention) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Parallel heads, different subspaces |
| 9 | [Causal Mask](docs/01-attention.md#causal-mask) | 🔥🔥🔥 | ⭐⭐ | Lower triangular, prevent future peeking |
| 10 | [GQA](docs/01-attention.md#grouped-query-attention-gqa) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Q heads > KV heads, LLaMA2 standard |
| 11 | [MQA](docs/01-attention.md#multi-query-attention-mqa) | 🔥🔥 | ⭐⭐⭐ | All Q share one KV |
| 12 | [Flash Attention](docs/01-attention.md#flash-attention-原理) | 🔥🔥 | ⭐⭐⭐⭐⭐ | Tiled computation, IO-aware, O(N) memory |
| 13 | [KV Cache](docs/01-attention.md#kv-cache) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Cache historical KV, avoid recomputation |
| 14 | [Cross Attention](docs/01-attention.md#multi-head-attention) | 🔥 | ⭐⭐⭐ | Q from decoder, KV from encoder |

### 📏 Normalization → [View](docs/02-normalization.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 15 | [Layer Normalization](docs/02-normalization.md#layer-normalization) | 🔥🔥🔥 | ⭐⭐ | Normalize across features, Transformer standard |
| 16 | [RMS Normalization](docs/02-normalization.md#rms-normalization) | 🔥🔥🔥 | ⭐⭐ | No mean, just RMS, LLaMA uses it |
| 17 | [Batch Normalization](docs/02-normalization.md#batch-normalization) | 🔥 | ⭐⭐ | Normalize across batch, CNN common |
| 18 | [Pre-Norm vs Post-Norm](docs/02-normalization.md#pre-norm-vs-post-norm) | 🔥🔥 | ⭐ | Pre-Norm more stable, modern LLM standard |

### 📍 Position Encoding → [View](docs/03-position-encoding.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 19 | [Sinusoidal PE](docs/03-position-encoding.md#sinusoidal-position-encoding) | 🔥 | ⭐⭐ | sin/cos fixed, original Transformer |
| 20 | [Learnable PE](docs/03-position-encoding.md#learnable-position-encoding) | 🔥 | ⭐ | Learnable embeddings, BERT/GPT |
| 21 | [RoPE](docs/03-position-encoding.md#rotary-position-embedding-rope) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Complex rotation, relative position, LLM standard |
| 22 | [ALiBi](docs/03-position-encoding.md#alibi) | 🔥🔥 | ⭐⭐⭐ | Linear bias in attention, good extrapolation |

### 🎲 Sampling Strategies → [View](docs/04-sampling.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 23 | [Greedy Decoding](docs/04-sampling.md#greedy-decoding) | 🔥 | ⭐ | Pick argmax each step |
| 24 | [Temperature Sampling](docs/04-sampling.md#temperature-sampling) | 🔥🔥🔥 | ⭐⭐ | `logits/T` controls randomness |
| 25 | [Top-k Sampling](docs/04-sampling.md#top-k-sampling) | 🔥🔥 | ⭐⭐ | Sample from top-k only |
| 26 | [Top-p Sampling](docs/04-sampling.md#top-p-nucleus-sampling) | 🔥🔥🔥 | ⭐⭐⭐ | Cumulative probability cutoff |
| 27 | [Beam Search](docs/04-sampling.md#beam-search) | 🔥🔥 | ⭐⭐⭐ | Keep k best sequences |

### 📉 Loss Functions → [View](docs/05-loss-functions.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 28 | [Cross Entropy Loss](docs/05-loss-functions.md#cross-entropy-loss) | 🔥🔥🔥 | ⭐⭐⭐ | `-log(p_true)`, classification standard |
| 29 | [LM Loss](docs/05-loss-functions.md#cross-entropy-loss) | 🔥🔥🔥 | ⭐⭐ | Autoregressive CE, next token prediction |
| 30 | [KL Divergence](docs/05-loss-functions.md#kl-divergence) | 🔥🔥 | ⭐⭐⭐ | Distribution difference, distillation/RLHF |
| 31 | [MSE Loss](docs/05-loss-functions.md#mse-loss) | 🔥 | ⭐ | `(y-ŷ)²`, regression |
| 32 | [Focal Loss](docs/05-loss-functions.md#focal-loss) | 🔥 | ⭐⭐⭐ | Down-weight easy samples |
| 33 | [SFT Loss](docs/05-loss-functions.md#sft-loss) | 🔥🔥 | ⭐⭐ | Masked CE, response only |
| 34 | [Reward Model Loss](docs/05-loss-functions.md#reward-model-loss) | 🔥🔥 | ⭐⭐⭐ | `-log σ(r_w - r_l)`, preference learning |
| 35 | [Contrastive Loss](docs/05-loss-functions.md#cross-entropy-loss) | 🔥 | ⭐⭐⭐ | Pull positives, push negatives |

### ⚡ Optimizers → [View](docs/06-optimizers.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 36 | [SGD](docs/06-optimizers.md#sgd) | 🔥 | ⭐ | Basic `w -= lr * grad` |
| 37 | [SGD + Momentum](docs/06-optimizers.md#sgd-with-momentum) | 🔥 | ⭐⭐ | Add momentum, faster convergence |
| 38 | [Adam](docs/06-optimizers.md#adam) | 🔥🔥🔥 | ⭐⭐⭐ | Adaptive LR, 1st & 2nd moments |
| 39 | [AdamW](docs/06-optimizers.md#adamw) | 🔥🔥🔥 | ⭐⭐⭐ | Decoupled weight decay, LLM standard |
| 40 | [LR Schedule](docs/06-optimizers.md#learning-rate-scheduler) | 🔥🔥 | ⭐⭐ | Warmup + Cosine/Linear decay |

### 🎮 Reinforcement Learning (RLHF) → [View](docs/07-reinforcement-learning.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 41 | [REINFORCE](docs/07-reinforcement-learning.md#reinforce) | 🔥 | ⭐⭐⭐ | Policy gradient `∇log π × R` |
| 42 | [GAE](docs/07-reinforcement-learning.md#gae) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Advantage estimation, bias-variance tradeoff |
| 43 | [PPO](docs/07-reinforcement-learning.md#ppo) | ��🔥🔥 | ⭐⭐⭐⭐⭐ | Clip ratio, RLHF core |
| 44 | [PPO-Clip](docs/07-reinforcement-learning.md#ppo) | ��🔥🔥 | ⭐⭐⭐⭐ | Clipped objective version |
| 45 | [DPO](docs/07-reinforcement-learning.md#dpo) | 🔥��🔥 | ⭐⭐⭐⭐ | Direct preference optimization, no RM |
| 46 | [GRPO](docs/07-reinforcement-learning.md#grpo) | 🔥🔥🔥 | ⭐⭐⭐⭐⭐ | Group relative policy, DeepSeek uses |
| 47 | [KL Penalty](docs/07-reinforcement-learning.md#ppo) | 🔥🔥 | ⭐⭐ | Prevent diverging from reference |
| 48 | [Reward Shaping](docs/07-reinforcement-learning.md#ppo) | 🔥 | ⭐⭐⭐ | Reward engineering, sparse → dense |

### 🚀 Efficient Training → [View](docs/08-efficient-training.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 49 | [LoRA](docs/08-efficient-training.md#lora) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Low-rank decomposition `W + BA` |
| 50 | [QLoRA](docs/08-efficient-training.md#lora) | 🔥🔥 | ⭐⭐⭐⭐ | LoRA + 4bit quantization |
| 51 | [Gradient Checkpointing](docs/08-efficient-training.md#gradient-checkpointing) | 🔥🔥 | ⭐⭐⭐ | Trade time for memory |
| 52 | [Mixed Precision](docs/08-efficient-training.md#mixed-precision-training) | 🔥🔥 | ⭐⭐⭐ | FP16/BF16, less memory, faster |
| 53 | [Gradient Accumulation](docs/08-efficient-training.md#gradient-accumulation) | 🔥🔥 | ⭐⭐ | Small batch simulates large batch |

### ⚡ Inference Optimization → [View](docs/09-inference-optimization.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 54 | [KV Cache](docs/09-inference-optimization.md#kv-cache) | 🔥🔥🔥 | ⭐⭐⭐⭐ | Cache KV, speed up autoregressive |
| 55 | [Paged Attention](docs/09-inference-optimization.md#pagedattention) | 🔥🔥 | ⭐⭐⭐⭐ | Paged KV management, vLLM core |
| 56 | [Speculative Decoding](docs/09-inference-optimization.md#speculative-decoding) | 🔥🔥 | ⭐⭐⭐⭐ | Small model drafts, large verifies |
| 57 | [Continuous Batching](docs/09-inference-optimization.md#continuous-batching) | 🔥🔥 | ⭐⭐⭐ | Dynamic batching, higher throughput |
| 58 | [Quantization](docs/09-inference-optimization.md#kv-cache) | 🔥🔥 | ⭐⭐⭐ | INT8/INT4, half+ memory |

### 🏗️ Transformer Architecture → [View](docs/10-transformer-architecture.md)

| # | Topic | Hot | Difficulty | One-liner |
|:---:|:---|:---:|:---:|:---|
| 59 | [Encoder-Only (BERT)](docs/10-transformer-architecture.md#transformer-概览) | 🔥 | ⭐⭐⭐ | Bidirectional, understanding tasks |
| 60 | [Decoder-Only (GPT)](docs/10-transformer-architecture.md#gpt-style-decoder-only) | 🔥🔥🔥 | ⭐⭐⭐ | Causal attention, generation, LLM standard |
| 61 | [Encoder-Decoder (T5)](docs/10-transformer-architecture.md#transformer-概览) | 🔥 | ⭐⭐⭐ | Seq2seq, translation/summarization |
| 62 | [FFN](docs/10-transformer-architecture.md#feed-forward-network) | 🔥🔥 | ⭐⭐ | 2-layer MLP, 4x expansion |
| 63 | [SwiGLU](docs/10-transformer-architecture.md#feed-forward-network) | 🔥🔥 | ⭐⭐⭐ | Gated FFN, LLaMA uses |

---

## 🔥 Hot Top 20

> Community-driven, updated hourly via GitHub Actions
>
> **Last updated**: 2026-04-18

| Rank | Topic | Category | Votes |
|:---:|:---|:---|:---:|
| 🥇 | [Scaled Dot-Product Attention](docs/01-attention.md#scaled-dot-product-attention) | Attention | 🔥 4 |
| 🥈 | [Gradient & Backprop](docs/00-llm-basics.md#梯度与反向传播) | Basics | 🔥 3 |
| 🥉 | [Linear Regression](docs/00-llm-basics.md#线性回归) | Basics | 🔥 3 |
| 4 | [Multi-Head Attention](docs/01-attention.md#multi-head-attention) | Attention | 🔥 3 |
| 5 | [Logistic Regression](docs/00-llm-basics.md#逻辑回归) | Basics | 🔥 2 |
| 6 | [BatchNorm](docs/02-normalization.md#batch-normalization) | Norm | 🔥 2 |
| 7 | [Cross Entropy](docs/05-loss-functions.md#cross-entropy-loss) | Loss | 🔥 2 |
| 8 | [LayerNorm](docs/02-normalization.md#layer-normalization) | Norm | 🔥 2 |
| 9 | [GQA](docs/01-attention.md#grouped-query-attention-gqa) | Attention | 🔥 1 |
| 10 | [RoPE](docs/03-position-encoding.md#rotary-position-embedding-rope) | Position | 🔥 1 |
| 11 | [DPO](docs/07-reinforcement-learning.md#dpo) | RL | 🔥 1 |
| 12 | [Causal Mask](docs/01-attention.md#causal-mask) | Attention | 🔥 1 |
| 13 | [Top-k](docs/04-sampling.md#top-k-sampling) | Sampling | 🔥 1 |
| 14 | [Top-p](docs/04-sampling.md#top-p-nucleus-sampling) | Sampling | 🔥 1 |
| 15 | [Beam Search](docs/04-sampling.md#beam-search) | Sampling | 🔥 1 |
| 16 | [Decoder-Only](docs/10-transformer-architecture.md#gpt-style-decoder-only) | Arch | 🔥 1 |
| 17 | [FFN](docs/10-transformer-architecture.md#feed-forward-network) | Arch | 🔥 1 |
| 18 | [GAE](docs/07-reinforcement-learning.md#gae) | RL | 🔥 1 |
| 19 | [GRPO](docs/07-reinforcement-learning.md#grpo) | RL | 🔥 1 |
| 20 | [KL Penalty](docs/07-reinforcement-learning.md#ppo) | RL | 🔥 1 |

---

## 🗳️ Vote Now

**Your interview experience matters!** Help calibrate real interview frequencies.

**👉 [Go to Voting Page](https://cdhx.github.io/LLM-Code-Hot-100) 👈**

- 🗳️ Vote for topics you've seen in interviews
- 🏆 Real-time leaderboard
- 💬 Share your experience

---

## 🤝 Contributing

Contributions welcome! New topics, bug fixes, documentation improvements.

1. **Fork** this repo
2. Create feature branch `git checkout -b feature/new-topic`
3. Commit changes `git commit -m 'Add: XXX'`
4. Push branch `git push origin feature/new-topic`
5. Submit **Pull Request**

---

## �� References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)

---

## ⭐ Star History

If this helps, please give it a star ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=cdhx/LLM-Code-Hot-100&type=Date)](https://star-history.com/#cdhx/LLM-Code-Hot-100&Date)

---

<div align="center">

**Made with ❤️ for LLM Interview Preparation**

*The Hot 100 for the LLM Era*

**#LLMHot100**

[Report Bug](https://github.com/cdhx/LLM-Code-Hot-100/issues) · [Request Feature](https://github.com/cdhx/LLM-Code-Hot-100/issues) · [Vote Now](https://cdhx.github.io/LLM-Code-Hot-100)

</div>
