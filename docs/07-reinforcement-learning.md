# 🎮 强化学习

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐⭐

RLHF 是 ChatGPT 成功的关键。PPO、DPO、GRPO 是面试热门话题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [REINFORCE](#reinforce)
- [GAE (Generalized Advantage Estimation)](#gae)
- [PPO (Proximal Policy Optimization)](#ppo)
- [DPO (Direct Preference Optimization)](#dpo)
- [GRPO (Group Relative Policy Optimization)](#grpo)
- [RLHF 全流程](#rlhf-全流程)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：RL 算法的核心区别在于**如何估计优势**和**如何更新策略**

| 方法 | 优势估计 | 策略更新 | 核心特点 | 使用场景 |
|:---|:---|:---|:---|:---|
| **REINFORCE** | R - baseline | 无约束 | 最基础 | 教学 |
| **PPO** | GAE | Clipped | 稳定 + 多轮更新 | RLHF |
| **DPO** | 内置 | 偏好直接转梯度 | 无需 RM/Critic | 离线偏好 |
| **GRPO** | 组内相对 | Clipped + KL | 无需 Critic | DeepSeek |

```python
# RL 算法的演进，理解从哪里来、解决什么问题

# REINFORCE: 最基础，梯度 = reward * logπ
loss = -R * logπ(a|s)                       # 问题: 方差大、不稳定

# PPO: 加了 GAE + Clipping
advantage = GAE(r, V)                       # 用 Critic 估计优势
loss = -min(ratio * A, clip(ratio) * A)     # 限制更新幅度

# DPO: 跳过 RM，直接从偏好学习
loss = -logσ(β * (logπ_w - logπ_l))      # 偏好对 -> 梯度

# GRPO: 组内比较，不需要 Critic
advantage = (r - mean(r)) / std(r)          # 组内相对优势
loss = PPO_loss + β * KL                    # 还是用 PPO 更新
```

> 🤔 **Q: PPO 和 DPO 有什么区别？什么时候用哪个？**
>
> | | PPO | DPO |
> |:---|:---|:---|
> | 需要 | RM + Critic + 采样 | 只需要偏好数据 |
> | 训练 | 复杂，多阶段 | 简单，一阶段 |
> | 探索 | 在线探索 | 无探索（离线） |
> | 适合 | 持续训练、需要探索 | 静态偏好数据 |
>
> 简单场景用 DPO，需要持续迭代/探索用 PPO。

---

## REINFORCE

### 🎯 核心思想

最基础的策略梯度算法：用奖励加权来估计梯度。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: torch.Tensor = None
) -> torch.Tensor:
    """
    REINFORCE 策略梯度损失
    
    Args:
        log_probs: [batch, seq_len] 每个 action 的 log 概率
        rewards: [batch] 或 [batch, seq_len] 奖励
        baseline: [batch] 基线（减少方差）
    
    Returns:
        loss: 策略梯度损失（负号因为优化器是最小化）
    """
    # 如果 rewards 是整个序列的奖励，扩展到每个 step
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1).expand_as(log_probs)
    
    # 减去 baseline 减少方差
    if baseline is not None:
        advantages = rewards - baseline.unsqueeze(1)
    else:
        advantages = rewards
    
    # REINFORCE: -E[log π(a|s) * A]
    # 负号是因为我们要最大化 reward，但 optimizer 是最小化 loss
    policy_loss = -(log_probs * advantages.detach()).sum(dim=-1).mean()
    
    return policy_loss


class REINFORCEWithBaseline:
    """
    带基线的 REINFORCE
    
    基线通常用 Value Network 估计，减少梯度方差
    """
    
    def __init__(self, policy_net, value_net, lr=1e-3):
        self.policy = policy_net
        self.value = value_net
        
        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    def compute_returns(self, rewards, gamma=0.99):
        """计算折扣回报"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def update(self, states, actions, rewards):
        """
        states: [batch, state_dim]
        actions: [batch]
        rewards: [batch] 每个 step 的奖励
        """
        # 计算回报
        returns = self.compute_returns(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算 value baseline
        values = self.value(states).squeeze()
        
        # 计算优势
        advantages = returns - values.detach()
        
        # Policy loss
        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # 更新 policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新 value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
```

---

## GAE

### 🎯 核心思想

GAE (Generalized Advantage Estimation) 在 TD(0) 和 Monte Carlo 之间做权衡：
- **λ = 0**: TD(0)，高偏差低方差
- **λ = 1**: Monte Carlo，低偏差高方差

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(TD error)}$$

> 🤔 **Q: λ 是什么意思？为什么通常取 0.95？**
>
> λ 控制"看多远"：
> - λ=0: 只用当前 step 的 TD error，偏差大但方差小
> - λ=1: 用整个 trajectory，方差大但无偏
>
> 0.95 是经验值，偏向于使用更多未来信息，同时保持稳定。

### 📝 实现代码

```python
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation 🔥 PPO 核心组件
    
    Args:
        rewards: [T] 每步奖励
        values: [T+1] 价值估计（包含 next_value）
        dones: [T] 是否终止
        gamma: 折扣因子（通常 0.99）
        lam: GAE lambda（通常 0.95，平衡偏差-方差）
    
    Returns:
        advantages: [T] 优势估计
        returns: [T] 回报（用于更新 value function）
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0
    
    # 从后往前计算（递归展开）
    for t in reversed(range(T)):
        # 终止状态：下一个状态的值为 0
        next_value = values[t + 1] * (1 - dones[t])
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE 递归: A_t = δ_t + γλ * A_{t+1}
        # (1 - dones[t]) 处理 episode 边界
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 回报 = 优势 + 价值（用于训练 value network）
    returns = advantages + values[:-1]
    
    # 标准化优势（稳定训练）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def compute_gae_vectorized(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    向量化 GAE（更快，适合大批量）
    
    rewards, values, dones: [batch, T]
    """
    batch_size, T = rewards.shape
    
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=rewards.device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[:, -1]  # 使用最后的 value
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
    
    returns = advantages + values
    
    # 标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns
```

---

## PPO

### 🎯 核心思想

PPO 是 RLHF 的主力算法。核心创新是 **Clipped Surrogate Objective**：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率。

> 🤔 **Q: 为什么用 min 而不是直接用 clipped？**
>
> 关键洞察：只有当 ratio “走过头” 时才需要 clip！
>
> - A > 0（好 action）: 我们想增大 ratio，但 ratio 已经 > 1+ε 时，clipped 更小，取 min 限制上升
> - A < 0（坏 action）: 我们想减小 ratio，但 ratio 已经 < 1-ε 时，clipped 更大，取 min 限制下降
>
> 总之：不管哪个方向，都防止更新过大。

### 📝 实现代码

```python
class PPO:
    """
    Proximal Policy Optimization 🔥🔥🔥 RLHF 核心算法
    """
    
    def __init__(
        self,
        policy_net,
        value_net,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy_net
        self.value = value_net
        
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 可以用一个 optimizer（如果 policy 和 value 共享参数）
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=lr
        )
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        计算 PPO 损失
        
        Returns:
            total_loss: 总损失
            info: 包含各部分损失的字典
        """
        # ========== 1. Policy Loss (Clipped Surrogate) ==========
        # 计算新的 log prob
        new_log_probs, entropy = self.policy.evaluate(states, actions)
        
        # 重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        # 取 min 是因为：
        # - 如果 advantage > 0，我们想增加这个 action 的概率
        #   但不希望 ratio 增加太多（被 clip 限制）
        # - 如果 advantage < 0，我们想减少概率
        #   但不希望 ratio 减少太多
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # ========== 2. Value Loss ==========
        values = self.value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # ========== 3. Entropy Bonus ==========
        # 鼓励探索，防止过早收敛
        entropy_loss = -entropy.mean()
        
        # ========== 总损失 ==========
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'approx_kl': ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
        }
        
        return total_loss, info
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        num_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        PPO 更新一轮
        """
        # 计算 value
        with torch.no_grad():
            values = self.value(states).squeeze()
            # 添加 next_value
            next_value = self.value(states[-1:]).squeeze()
            values_with_next = torch.cat([values, next_value])
        
        # 计算 GAE
        advantages, returns = compute_gae(rewards, values_with_next, dones, self.gamma, self.lam)
        
        # 多轮更新（PPO 可以重复使用采样数据）
        dataset_size = len(states)
        
        for epoch in range(num_epochs):
            # Mini-batch 更新
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                loss, info = self.compute_loss(
                    states[batch_indices],
                    actions[batch_indices],
                    old_log_probs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices]
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
        
        return info
```

---

## DPO

### 🎯 核心思想

DPO 跳过 Reward Model，直接从偏好数据优化 Policy：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**直觉**：让模型更偏好 chosen response，同时不偏离 reference model 太远。

> 🤔 **Q: DPO 的 β 是什么意思？怎么选？**
>
> β 控制"离 reference 多远"：
> - β 大: 惩罚大，不能偏离 ref 太多（保守）
> - β 小: 惩罚小，可以大胆偏离 ref（激进）
>
> 通常 β ∈ [0.1, 0.5]，需要调参。

### 📝 实现代码

```python
class DPOTrainer:
    """
    Direct Preference Optimization 🔥🔥🔥
    
    优点：
    - 不需要训练 Reward Model
    - 不需要 PPO 的复杂采样
    - 训练更稳定
    
    缺点：
    - 依赖高质量偏好数据
    - 没有在线探索，可能过拟合
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,  # 冻结的参考模型
        beta: float = 0.1,
        lr: float = 1e-6
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta
        
        # 冻结 reference model
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr
        )
    
    def get_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列的 log 概率
        
        input_ids: [batch, seq_len] prompt + response
        labels: [batch, seq_len] 只有 response 部分有效（prompt 部分为 -100）
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # Shift: logits[:, :-1] 预测 labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 取出真实 token 的 log prob
        # [batch, seq_len-1]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask 掉 padding 和 prompt 部分
        mask = (shift_labels != -100).float()
        
        # 求和得到序列 log prob
        seq_log_probs = (token_log_probs * mask).sum(dim=-1)
        
        return seq_log_probs
    
    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        """
        计算 DPO 损失
        
        batch 包含:
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # ========== Policy model log probs ==========
        policy_chosen_logps = self.get_log_probs(
            self.policy,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels']
        )
        policy_rejected_logps = self.get_log_probs(
            self.policy,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels']
        )
        
        # ========== Reference model log probs ==========
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels']
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels']
            )
        
        # ========== DPO Loss ==========
        # log(π/π_ref) 差值
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO loss = -log(sigmoid(β * (pi_logratio - ref_logratio)))
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # 计算指标
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        info = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_rewards': chosen_rewards.mean().item(),
            'rejected_rewards': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }
        
        return loss, info
    
    def train_step(self, batch):
        loss, info = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return info
```

---

## GRPO

### 🎯 核心思想

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的算法：
- **去掉 Critic**: 用组内相对奖励代替 value baseline
- **组内比较**: 同一 prompt 采样多个 response，相互比较

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)} \quad \text{(组内标准化优势)}$$

> 🤔 **Q: GRPO 为什么不需要 Critic？这不会不稳定吗？**
>
> GRPO 的精妙在于：同一个 prompt 的多个 response 形成"对照组"。
>
> 组内标准化 `(r - mean) / std` 相当于自动的 baseline，不需要学一个 Critic。
>
> 为什么稳定？因为比较的是同一个 prompt 的 response，变量少很多。

### 📝 实现代码

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization 🔥🔥🔥 DeepSeek 使用
    
    核心思想：
    1. 对同一个 prompt 采样 G 个不同的 response
    2. 用组内相对奖励作为优势，不需要 Critic
    3. 用 PPO 的 clipping 更新 policy
    
    优势：
    - 比 PPO 简单（不需要 value network）
    - 比 DPO 更在线（可以探索）
    - 训练更稳定
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,
        group_size: int = 4,  # 每个 prompt 采样几个 response
        beta: float = 0.04,   # KL 惩罚系数
        clip_eps: float = 0.2,
        lr: float = 1e-6
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.group_size = group_size
        self.beta = beta
        self.clip_eps = clip_eps
        
        # 冻结 reference
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    
    def compute_rewards(self, prompts, responses):
        """
        计算奖励（可以是规则或 Reward Model）
        
        例如：数学题检查答案是否正确
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            # 这里用简化的规则奖励
            # 实际可以用 Reward Model 或执行代码验证
            reward = self.rule_based_reward(prompt, response)
            rewards.append(reward)
        return torch.tensor(rewards)
    
    def rule_based_reward(self, prompt, response):
        """示例：基于规则的奖励"""
        # 实际中可以：执行代码检查输出、数学验证、格式检查等
        return 0.0  # placeholder
    
    def compute_loss(
        self,
        prompts,           # [batch_size] 提示
        responses_list,    # [[batch_size] * G] G 组回答
        rewards_tensor,    # [batch_size, G] 奖励
        old_log_probs_list  # [[batch_size] * G] 采样时的 log prob
    ):
        """
        GRPO 核心损失计算
        """
        batch_size = len(prompts)
        G = self.group_size
        
        # ========== 1. 计算组内相对优势 ==========
        # rewards: [batch, G]
        mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)  # [batch, 1]
        std_rewards = rewards_tensor.std(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        
        # 标准化优势：不需要 Critic！
        advantages = (rewards_tensor - mean_rewards) / std_rewards  # [batch, G]
        
        # ========== 2. 计算 PPO-style clipped loss ==========
        total_loss = 0
        
        for g in range(G):
            # 当前 policy 的 log prob
            new_log_probs = self.get_log_probs(
                self.policy,
                prompts,
                responses_list[g]
            )
            
            # Reference 的 log prob
            with torch.no_grad():
                ref_log_probs = self.get_log_probs(
                    self.ref,
                    prompts,
                    responses_list[g]
                )
            
            # 重要性采样比率
            old_log_probs = old_log_probs_list[g]
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate
            adv = advantages[:, g]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL 惩罚
            kl = (new_log_probs - ref_log_probs).mean()
            kl_penalty = self.beta * kl
            
            total_loss += policy_loss + kl_penalty
        
        return total_loss / G
    
    def get_log_probs(self, model, prompts, responses):
        """计算 response 的 log prob"""
        # 简化实现
        # 实际需要正确处理 tokenization 和 padding
        pass
    
    def train_step(self, prompts):
        """
        GRPO 训练一步
        
        1. 对每个 prompt 采样 G 个 response
        2. 计算奖励
        3. 计算组内优势
        4. PPO 更新
        """
        # 采样
        responses_list = []
        old_log_probs_list = []
        
        with torch.no_grad():
            for _ in range(self.group_size):
                responses, log_probs = self.policy.sample(prompts)
                responses_list.append(responses)
                old_log_probs_list.append(log_probs)
        
        # 计算奖励
        all_rewards = []
        for responses in responses_list:
            rewards = self.compute_rewards(prompts, responses)
            all_rewards.append(rewards)
        rewards_tensor = torch.stack(all_rewards, dim=1)  # [batch, G]
        
        # 计算并优化
        loss = self.compute_loss(
            prompts,
            responses_list,
            rewards_tensor,
            old_log_probs_list
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## RLHF 全流程

### 📝 完整流程代码

```python
class RLHFPipeline:
    """
    RLHF 完整流程
    
    Stage 1: SFT (Supervised Fine-Tuning)
    Stage 2: Reward Model Training
    Stage 3: PPO/GRPO/DPO Training
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    # ========== Stage 1: SFT ==========
    def sft_train(self, sft_dataset, epochs=3):
        """
        监督微调：在高质量数据上训练
        """
        self.policy = self.base_model
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            for batch in sft_dataset:
                loss = self.compute_sft_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.policy
    
    def compute_sft_loss(self, batch):
        """SFT Loss = Next Token Prediction"""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        outputs = self.policy(input_ids)
        logits = outputs.logits
        
        # Shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss
    
    # ========== Stage 2: Reward Model ==========
    def train_reward_model(self, preference_dataset, epochs=1):
        """
        训练 Reward Model
        
        输入: (prompt, chosen, rejected) 三元组
        目标: R(chosen) > R(rejected)
        """
        self.reward_model = RewardModel(self.base_model.config)
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            for batch in preference_dataset:
                chosen_rewards = self.reward_model(batch['chosen_input_ids'])
                rejected_rewards = self.reward_model(batch['rejected_input_ids'])
                
                # Bradley-Terry Loss
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.reward_model
    
    # ========== Stage 3: RL Training ==========
    def rl_train_ppo(self, prompts_dataset, epochs=1):
        """PPO 训练"""
        ref_model = copy.deepcopy(self.policy)
        for param in ref_model.parameters():
            param.requires_grad = False
        
        ppo = PPO(self.policy, self.value_net)
        
        for epoch in range(epochs):
            for prompts in prompts_dataset:
                # 1. 生成 response
                responses, old_log_probs = self.policy.sample(prompts)
                
                # 2. 计算 reward
                rewards = self.reward_model(prompts, responses)
                
                # 3. KL 惩罚
                ref_log_probs = ref_model.log_prob(prompts, responses)
                kl_penalty = self.kl_coef * (old_log_probs - ref_log_probs)
                rewards = rewards - kl_penalty
                
                # 4. PPO 更新
                ppo.update(prompts, responses, old_log_probs, rewards)
    
    def rl_train_dpo(self, preference_dataset, epochs=1):
        """DPO 训练（更简单）"""
        ref_model = copy.deepcopy(self.policy)
        dpo = DPOTrainer(self.policy, ref_model)
        
        for epoch in range(epochs):
            for batch in preference_dataset:
                dpo.train_step(batch)


class RewardModel(nn.Module):
    """Reward Model 结构"""
    
    def __init__(self, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.reward_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]
        
        # 输出标量奖励
        reward = self.reward_head(last_hidden).squeeze(-1)
        
        return reward
```

---

## 面试追问汇总

### PPO 相关

| 问题 | 答案 |
|:---|:---|
| PPO 的 clip 有什么作用 | 限制策略更新幅度，防止 reward hacking |
| 为什么需要 GAE | 平衡 bias-variance，比 Monte Carlo 方差更小 |
| PPO 比 TRPO 好在哪 | 更简单，一阶优化而非二阶 |

### DPO vs PPO

| 方面 | PPO | DPO |
|:---|:---|:---|
| 需要 Reward Model | 是 | 否 |
| 需要 Critic | 是 | 否 |
| 训练稳定性 | 较差 | 较好 |
| 在线探索 | 是 | 否 |
| 数据要求 | 可以复用 | 需要偏好对 |

### GRPO 相关

| 问题 | 答案 |
|:---|:---|
| GRPO 为什么不需要 Critic | 用组内相对奖励代替 value baseline |
| GRPO 的优势 | 比 PPO 简单，比 DPO 更在线 |
| group_size 选多大 | 通常 4-8 |

### 代码追问

```python
# Q: DPO 的 beta 有什么作用？
"""
A: beta 控制 KL 惩罚强度
- beta 大: policy 更接近 reference，更保守
- beta 小: policy 可以偏离更多，更激进
通常取 0.1-0.5
"""

# Q: PPO 为什么可以重复使用采样数据？
"""
A: 因为用了重要性采样 (ratio = π_new / π_old)
可以用旧策略的样本估计新策略的期望
但 ratio 偏离太大时估计不准，所以要 clip
"""
```

---

## 🔗 相关题目

- [KL Divergence](05-loss-functions.md#kl-divergence) - RL 中的 KL 惩罚
- [Reward Model Loss](05-loss-functions.md#reward-model-loss) - RM 训练
- [SFT Loss](05-loss-functions.md#sft-loss) - RLHF 第一阶段
