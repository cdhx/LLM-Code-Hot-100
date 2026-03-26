# 📉 损失函数

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐

损失函数是模型训练的核心。Cross Entropy 的数值稳定性实现是必考题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [Cross Entropy Loss](#cross-entropy-loss)
- [Softmax 数值稳定性](#softmax-数值稳定性)
- [Focal Loss](#focal-loss)
- [KL Divergence](#kl-divergence)
- [MSE Loss](#mse-loss)
- [SFT Loss (LLM 训练)](#sft-loss)
- [Reward Model Loss](#reward-model-loss)
- [Label Smoothing](#label-smoothing)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：损失函数的核心区别在于**衡量什么**

| 方法 | 衡量内容 | 核心公式 | 使用场景 |
|:---|:---|:---|:---|
| **Cross Entropy** | 预测与真实的差异 | `-log(p_target)` | 分类、LLM |
| **Focal Loss** | CE + 难样本加权 | `(1-p)² * CE` | 类别不平衡 |
| **KL Divergence** | 两个分布的差异 | `p * log(p/q)` | 知识蒸馏、RLHF |
| **MSE** | 预测与真实的差的平方 | `(y - ŷ)²` | 回归 |
| **Reward Model Loss** | 偏好对的正确排序 | `-logσ(r_w - r_l)` | RLHF |

```python
# LLM 训练中的损失函数

# SFT: 简单的 Cross Entropy
loss = -log(p(next_token | context))      # 每个 token 的 CE

# RM: 学习偏好排序
loss = -logσ(r_chosen - r_rejected)        # Bradley-Terry 模型

# DPO: 直接从偏好中学习
loss = -logσ(β * (logπ - logπ_ref))      # 无需单独的 RM

# PPO: 策略梯度 + KL 惩罚
loss = -advantage * ratio + β * KL(π || π_ref)  # 限制更新幅度
```

> 🤔 **Q: 为什么 LLM 用 Cross Entropy 而不是 MSE？**
>
> 1. LLM 是分类任务（预测下一个 token），不是回归
> 2. CE 的梯度比 MSE 更合理：当预测错误时 CE 梯度更大、更新更快
> 3. CE 跟 softmax 配合天然，softmax 的导数刚好是预测-真实

---

## Cross Entropy Loss

### 🎯 核心思想

交叉熵衡量两个概率分布的差异。对于分类任务：

$$\text{CE}(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)$$

当 $y$ 是 one-hot 编码时（真实类别为 $c$）：

$$\text{CE} = -\log(\hat{y}_c) = -\log\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right)$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean',
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Cross Entropy Loss 手动实现（数值稳定版）
    
    Args:
        logits: [batch, num_classes] 模型原始输出（未经 softmax）
        targets: [batch] 真实类别索引
        reduction: 'mean', 'sum', 'none'
        ignore_index: 忽略的标签（如 padding）
    
    Returns:
        loss: 标量或 [batch]（取决于 reduction）
    """
    batch_size, num_classes = logits.shape
    
    # ========== 数值稳定性处理 ==========
    # 直接计算 softmax 会导致 exp(logits) 溢出
    # 技巧: softmax(x) = softmax(x - max(x))
    # 这样最大的 logit 变成 0，exp(0) = 1，不会溢出
    
    # Step 1: 减去最大值（数值稳定）
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max  # 最大值变成 0
    
    # Step 2: 计算 log_softmax
    # log_softmax(x) = x - log(sum(exp(x)))
    # 但 log(sum(exp(x))) 可以用 logsumexp 更稳定地计算
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
    log_softmax = logits_stable - log_sum_exp  # [batch, num_classes]
    
    # Step 3: 取出真实类别的 log 概率（负对数似然）
    # gather: 从 log_softmax 中按 targets 索引取值
    nll = -log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # nll: [batch]
    
    # Step 4: 处理 ignore_index
    if ignore_index is not None:
        mask = targets != ignore_index
        nll = nll * mask.float()
    
    # Step 5: Reduction
    if reduction == 'none':
        return nll
    elif reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        if ignore_index is not None:
            return nll.sum() / mask.sum().clamp(min=1)
        return nll.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def cross_entropy_simple(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy 极简版（面试快速手写）
    """
    # 方法1: 分解为 log_softmax + nll_loss
    log_probs = F.log_softmax(logits, dim=-1)  # 内部已做数值稳定
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()
    
    # 方法2: 更简洁
    # return F.cross_entropy(logits, targets)


# ==================== 验证代码 ====================
def test_cross_entropy():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    
    my_loss = cross_entropy_loss(logits, targets)
    torch_loss = F.cross_entropy(logits, targets)
    
    print(f"我的实现: {my_loss:.6f}")
    print(f"PyTorch:  {torch_loss:.6f}")
    print(f"误差:     {abs(my_loss - torch_loss):.8f}")
    
    assert abs(my_loss - torch_loss) < 1e-5
    print("✅ Cross Entropy 测试通过!")


if __name__ == "__main__":
    test_cross_entropy()
```

> 🤔 **Q: 为什么用 `gather` 而不是直接索引 `log_softmax[targets]`？**
>
> 因为需要批量处理！`log_softmax[targets]` 对 2D tensor 不行。
>
> `gather(dim=-1, index=targets.unsqueeze(-1))` 正确地从每个样本的分布中取出目标类别的概率。

---

## Softmax 数值稳定性

### 🎯 核心问题

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**问题**：当 $x_i$ 很大（如 100）时，$e^{100}$ 会溢出（float32 最大约 $3.4 \times 10^{38}$）

> 🤔 **Q: 为什么减去 max 就能解决溢出问题？**
>
> 数学上：`softmax(x) = softmax(x - c)`，任意常数 c 不改变结果。
>
> 选 `c = max(x)` 后，最大的 `x_i - c = 0`，`exp(0) = 1`，不会溢出！
>
> 其他的 `x_j - c < 0`，`exp(负数) < 1`，也不会溢出。

### 📝 实现代码

```python
def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """
    朴素 Softmax（不稳定，仅作对比）
    """
    exp_x = torch.exp(x)  # 可能溢出！
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def softmax_stable(x: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 Softmax
    
    技巧: softmax(x) = softmax(x - c)，任意常数 c
    选择 c = max(x)，使得最大的 exp 参数为 0
    """
    # Step 1: 减去最大值
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Step 2: 安全的 exp
    exp_x = torch.exp(x_shifted)  # 最大值的 exp 是 1，不会溢出
    
    # Step 3: 归一化
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def log_softmax_stable(x: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 Log-Softmax
    
    log_softmax(x) = x - log(sum(exp(x)))
                   = x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=-1, keepdim=True))
    
    return x_shifted - log_sum_exp


# ==================== 测试溢出情况 ====================
def test_overflow():
    # 大数值测试
    x_large = torch.tensor([100.0, 200.0, 300.0])
    
    print("=== 溢出测试 ===")
    print(f"输入: {x_large}")
    
    try:
        naive_result = softmax_naive(x_large)
        print(f"朴素 Softmax: {naive_result}")  # 会有 inf/nan
    except Exception as e:
        print(f"朴素 Softmax 失败: {e}")
    
    stable_result = softmax_stable(x_large)
    print(f"稳定 Softmax: {stable_result}")  # 正确结果
    
    pytorch_result = F.softmax(x_large, dim=-1)
    print(f"PyTorch:      {pytorch_result}")


if __name__ == "__main__":
    test_overflow()
```

输出：
```
=== 溢出测试 ===
输入: tensor([100., 200., 300.])
朴素 Softmax: tensor([0., 0., nan])
稳定 Softmax: tensor([0.0000e+00, 0.0000e+00, 1.0000e+00])
PyTorch:      tensor([0.0000e+00, 0.0000e+00, 1.0000e+00])
```

---

## Focal Loss

### 🎯 核心思想

解决类别不平衡问题：降低易分类样本的权重，关注难分类样本。

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $(1 - p_t)^\gamma$：调制因子，$p_t$ 接近 1 时权重降低
- $\gamma$：聚焦参数，通常取 2

### 📝 实现代码

```python
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss for Dense Object Detection
    
    Args:
        logits: [batch, num_classes] 模型输出
        targets: [batch] 真实标签
        alpha: 平衡正负样本的权重
        gamma: 聚焦参数（gamma=0 时退化为 CE）
        reduction: 'mean', 'sum', 'none'
    """
    # Step 1: 计算 CE
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Step 2: 计算 pt（正确类别的预测概率）
    probs = F.softmax(logits, dim=-1)
    pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Step 3: 计算 focal weight
    # (1 - pt)^gamma: 易分类样本 pt->1，权重->0
    focal_weight = (1 - pt) ** gamma
    
    # Step 4: 应用权重
    focal_loss = focal_weight * ce_loss
    
    # Step 5: 类别平衡权重（可选）
    if alpha is not None:
        # 简化处理：正类用 alpha，负类用 1-alpha
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(alpha, device=logits.device),
            torch.tensor(1 - alpha, device=logits.device)
        )
        focal_loss = alpha_t * focal_loss
    
    # Step 6: Reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = None,  # [num_classes]
    gamma: float = 2.0
) -> torch.Tensor:
    """
    多分类 Focal Loss
    
    alpha: 每个类别的权重，可用于处理类别不平衡
    """
    num_classes = logits.size(-1)
    
    # CE loss
    ce = F.cross_entropy(logits, targets, reduction='none')
    
    # pt
    probs = F.softmax(logits, dim=-1)
    pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Focal weight
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weight
    if alpha is not None:
        alpha_t = alpha[targets]
        return (alpha_t * focal_weight * ce).mean()
    
    return (focal_weight * ce).mean()
```

---

## KL Divergence

### 🎯 核心思想

KL 散度衡量两个概率分布的差异：

$$D_{KL}(P \| Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}$$

**注意**：KL 散度不对称！$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

### 📝 实现代码

```python
def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    KL Divergence: D_KL(P || Q)
    
    Args:
        p: [batch, num_classes] 目标分布（概率）
        q: [batch, num_classes] 预测分布（概率）
        reduction: 'batchmean', 'sum', 'none'
    """
    eps = 1e-10
    
    # 确保概率有效
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    
    # KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
    kl = p * (torch.log(p) - torch.log(q))
    kl = kl.sum(dim=-1)  # 对类别求和
    
    if reduction == 'batchmean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl


def kl_divergence_from_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    从 logits 计算 KL 散度（更稳定）
    
    用 log_softmax 避免 softmax 后再 log 的精度损失
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    
    p = torch.exp(log_p)
    
    # KL = sum(p * (log_p - log_q))
    kl = (p * (log_p - log_q)).sum(dim=-1)
    
    if reduction == 'batchmean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon Divergence（对称版 KL）
    
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    其中 M = (P + Q) / 2
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, 'batchmean') + 0.5 * kl_divergence(q, m, 'batchmean')


# ==================== RL 中的 KL 惩罚 ====================
def kl_penalty_for_rl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor
) -> torch.Tensor:
    """
    RLHF/PPO 中的 KL 惩罚项
    
    用于防止 policy 偏离 reference model 太远
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    
    policy_probs = torch.exp(policy_log_probs)
    
    # D_KL(policy || ref)
    kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    
    return kl.mean()
```

---

## MSE Loss

### 🎯 核心思想

均方误差，用于回归任务：

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### 📝 实现代码

```python
def mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Mean Squared Error Loss
    
    Args:
        predictions: 预测值
        targets: 目标值
        reduction: 'mean', 'sum', 'none'
    """
    squared_diff = (predictions - targets) ** 2
    
    if reduction == 'none':
        return squared_diff
    elif reduction == 'sum':
        return squared_diff.sum()
    elif reduction == 'mean':
        return squared_diff.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error"""
    return torch.sqrt(mse_loss(predictions, targets, reduction='mean'))


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (L1 Loss)"""
    return torch.abs(predictions - targets).mean()


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    """
    Huber Loss: MSE 和 MAE 的平滑组合
    
    |error| < delta: MSE（平滑）
    |error| >= delta: MAE（抗离群点）
    """
    error = predictions - targets
    abs_error = torch.abs(error)
    
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    
    return (0.5 * quadratic ** 2 + delta * linear).mean()
```

---

## SFT Loss

### 🎯 核心思想

Supervised Fine-Tuning Loss：LLM 的标准训练损失，本质是 **next token prediction**。

### 📝 实现代码

```python
def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    SFT Loss: Language Model 的标准训练损失
    
    Args:
        logits: [batch, seq_len, vocab_size] 模型输出
        labels: [batch, seq_len] 目标 token（通常是 input_ids 左移一位）
        ignore_index: 忽略的标签（如 padding、prompt 部分）
    
    Returns:
        loss: 平均 loss
    """
    # 重塑为 2D
    # logits: [batch * seq_len, vocab_size]
    # labels: [batch * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Cross Entropy（内部处理 ignore_index）
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


def sft_loss_with_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor
) -> torch.Tensor:
    """
    带 mask 的 SFT Loss
    
    loss_mask: [batch, seq_len] 1 表示计算 loss，0 表示忽略
    
    常见用法：
    - 只对 response 部分计算 loss（忽略 prompt）
    - 忽略 padding
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Shift: labels 应该是 logits 预测的下一个 token
    # logits[:, :-1] 预测 labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()
    
    # 计算每个 token 的 loss
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='none'
    ).view(batch_size, -1)
    
    # 应用 mask
    masked_loss = loss_per_token * shift_mask
    
    # 平均（只除以有效 token 数）
    return masked_loss.sum() / shift_mask.sum().clamp(min=1)


# ==================== 使用示例 ====================
def sft_example():
    """SFT 训练示例"""
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    # 模拟模型输出
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 模拟标签（通常是 input_ids 本身，模型预测下一个 token）
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 模拟 mask（前 3 个是 prompt，不计算 loss）
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :3] = 0  # prompt 部分
    
    loss = sft_loss_with_mask(logits, labels, loss_mask)
    print(f"SFT Loss: {loss:.4f}")
```

---

## Reward Model Loss

### 🎯 核心思想

Reward Model 学习人类偏好：给定同一 prompt 的两个 response，判断哪个更好。

$$\mathcal{L} = -\log\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $y_w$ 是 chosen（更好的），$y_l$ 是 rejected（更差的）。

> 🤔 **Q: 为什么用 `r_chosen - r_rejected` 而不是直接让 chosen 得分高？**
>
> 这是 Bradley-Terry 模型：我们关心的是**相对质量**，不是绝对分数。
>
> 如果直接让 chosen 得分高，模型会让所有分数越来越高（分数膨胀）。
>
> 用差值后，模型只需要学会“谁更好”，不用关心绝对分数。

### 📝 实现代码

```python
def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """
    Reward Model Loss (Bradley-Terry Model)
    
    Args:
        chosen_rewards: [batch] chosen response 的奖励分数
        rejected_rewards: [batch] rejected response 的奖励分数
    
    Returns:
        loss: 标量
    """
    # Bradley-Terry: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    # Loss = -log(P) = -log(sigmoid(r_chosen - r_rejected))
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    
    return loss.mean()


def reward_model_loss_with_margin(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.0
) -> torch.Tensor:
    """
    带 margin 的 Reward Model Loss
    
    要求 chosen 比 rejected 高出至少 margin
    """
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards - margin)
    return loss.mean()


class RewardModel(torch.nn.Module):
    """
    Reward Model 示例结构
    """
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model  # 预训练 LLM
        self.reward_head = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # 获取最后一个 token 的 hidden state
        outputs = self.base(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # 取最后一个非 padding token
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        last_token_hidden = last_hidden[
            torch.arange(batch_size),
            seq_lengths
        ]
        
        # 输出标量奖励
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward


def train_reward_model_step(
    model,
    chosen_input_ids,
    chosen_attention_mask,
    rejected_input_ids,
    rejected_attention_mask,
    optimizer
):
    """Reward Model 训练一步"""
    # 前向传播
    chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
    rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
    
    # 计算 loss
    loss = reward_model_loss(chosen_rewards, rejected_rewards)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss.item(), accuracy.item()
```

---

## Label Smoothing

### 🎯 核心思想

将 hard label（one-hot）软化，防止模型过于自信。

$$y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{hard}} + \frac{\epsilon}{K}$$

### 📝 实现代码

```python
def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Label Smoothing Cross Entropy
    
    Args:
        logits: [batch, num_classes]
        targets: [batch] hard labels
        smoothing: 平滑系数（通常 0.1）
    """
    num_classes = logits.size(-1)
    
    # 创建 soft labels
    # confidence = 1 - smoothing (真实类别)
    # smoothing / num_classes (其他类别)
    confidence = 1.0 - smoothing
    smooth_value = smoothing / num_classes
    
    # 创建 soft target
    soft_targets = torch.full_like(logits, smooth_value)
    soft_targets.scatter_(
        dim=-1,
        index=targets.unsqueeze(-1),
        value=confidence + smooth_value
    )
    
    # KL divergence (等价于 cross entropy with soft targets)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    
    return loss.mean()
```

---

## 面试追问汇总

### 数值稳定性

| 问题 | 答案 |
|:---|:---|
| 为什么要用 log_softmax | 避免先 softmax 再 log 的精度损失 |
| Cross Entropy 为什么减 max | 防止 exp 溢出 |
| 什么时候会出现 NaN | exp 溢出、log(0)、除零 |

### 损失函数选择

| 问题 | 答案 |
|:---|:---|
| Focal Loss 适用场景 | 类别不平衡（如目标检测） |
| KL vs CE | KL 用于分布对齐，CE 用于分类 |
| Label Smoothing 的作用 | 防止过拟合，提高泛化 |

### 代码追问

```python
# Q: 这段代码有什么问题？
def buggy_ce(logits, targets):
    probs = torch.softmax(logits, dim=-1)
    loss = -torch.log(probs[range(len(targets)), targets])  # BUG!
    return loss.mean()

# A: 
# 1. log(softmax(x)) 应该用 log_softmax，精度更好
# 2. 没有处理 probs=0 的情况（会产生 -inf）
# 3. 没有数值稳定性处理
```

---

## 🔗 相关题目

- [DPO Loss](07-reinforcement-learning.md#dpo) - 基于偏好的损失
- [PPO Loss](07-reinforcement-learning.md#ppo) - RL 中的策略损失
- [GRPO Loss](07-reinforcement-learning.md#grpo) - 组相对策略优化
