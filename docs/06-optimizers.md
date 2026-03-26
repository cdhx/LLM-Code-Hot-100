# ⚡ 优化器

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐

优化器决定模型如何更新参数。Adam 是面试必考题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [SGD](#sgd)
- [SGD with Momentum](#sgd-with-momentum)
- [Adam](#adam)
- [AdamW](#adamw)
- [Learning Rate Scheduler](#learning-rate-scheduler)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：优化器的核心区别在于**如何修正梯度**

| 方法 | 核心思想 | 更新公式 | 特点 |
|:---|:---|:---|:---|
| **SGD** | 原始梯度下降 | `θ -= lr * g` | 简单但慢 |
| **Momentum** | 加速 + 减震 | `v = βv + g; θ -= lr * v` | 累积历史方向 |
| **Adam** | 自适应学习率 | `θ -= lr * m/√v` | 每个参数不同步长 |
| **AdamW** | Adam + 解耦权重衰减 | `θ *= (1-wd); θ -= lr * m/√v` | LLM 标配 |

```python
# 优化器的进化，每一步都在解决一个问题

# SGD: 最基础
θ -= lr * grad                         # 问题: 震荡、慢

# Momentum: 累积历史梯度
v = 0.9 * v + grad                       # 累积动量
θ -= lr * v                             # 解决: 加速 + 减震

# Adam: 每个参数自适应学习率
m = 0.9 * m + 0.1 * grad                 # 一阶矩（方向）
v = 0.999 * v + 0.001 * grad²            # 二阶矩（幅度）
θ -= lr * m / √v                       # 解决: 不同参数不同步长

# AdamW: 解耦 weight decay
θ *= (1 - lr * wd)                      # 先衰减权重
θ -= lr * m / √v                       # 再 Adam 更新
```

> 🤔 **Q: 为什么 LLM 用 AdamW 而不用 Adam？**
>
> Adam 的问题：weight decay 被加到 grad 上，会被自适应学习率缩放。
>
> 梯度大的参数 → adaptive lr 小 → weight decay 也被缩小，正则化不均匀。
>
> AdamW 把 weight decay 从 Adam 中抽出来，每个参数被一致地 decay。

---

## SGD

### 🎯 核心思想

最基础的梯度下降：沿着梯度反方向更新参数。

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$$

### 📝 实现代码

```python
import torch
import torch.nn as nn

class SGD:
    """
    Stochastic Gradient Descent
    """
    
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        """执行一步参数更新"""
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    # θ = θ - lr * grad
                    param -= self.lr * param.grad
    
    def zero_grad(self):
        """清空梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ==================== 使用示例 ====================
def test_sgd():
    # 简单线性回归
    torch.manual_seed(42)
    X = torch.randn(100, 2)
    y = X @ torch.tensor([2.0, 3.0]) + 1.0 + torch.randn(100) * 0.1
    
    # 模型参数
    w = torch.randn(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    optimizer = SGD([w, b], lr=0.1)
    
    for epoch in range(100):
        # 前向
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print(f"学到的权重: w={w.data}, b={b.data}")


if __name__ == "__main__":
    test_sgd()
```

---

## SGD with Momentum

### 🎯 核心思想

引入"动量"，积累历史梯度方向，加速收敛并减少震荡。

$$v_t = \gamma v_{t-1} + \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

**物理直觉**：像小球滚下山坡，有惯性。

### 📝 实现代码

```python
class SGDMomentum:
    """
    SGD with Momentum
    
    动量帮助：
    1. 加速：在一致方向上累积速度
    2. 减震：减少在局部最小值附近的震荡
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        
        # 初始化速度（每个参数一个）
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                v = self.velocities[i]
                
                # 更新速度: v = momentum * v + grad
                v.mul_(self.momentum).add_(grad)
                
                if self.nesterov:
                    # Nesterov: 先"预测"位置，再计算梯度
                    # θ = θ - lr * (momentum * v + grad)
                    param -= self.lr * (self.momentum * v + grad)
                else:
                    # 标准 Momentum
                    param -= self.lr * v
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

---

## Adam

### 🎯 核心思想

Adam = Adaptive Moment Estimation，结合了：
- **Momentum**（一阶矩：梯度均值）
- **RMSprop**（二阶矩：梯度方差）

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

> 🤔 **Q: 为什么除以 √v 而不是直接用 v？**
>
> v 是梯度平方的均值，量纲是 "梯度²"。
>
> 我们要除的是梯度的"幅度"，所以要开根号得到"梯度”量纲。
>
> 这样 m/√v 的量纲是“梯度/梯度=无量纲”，再乘 lr 才正确。

### 📝 实现代码

```python
class Adam:
    """
    Adam Optimizer 🔥 面试必考
    
    核心思想：
    1. m: 梯度的指数移动平均（一阶矩，类似 momentum）
    2. v: 梯度平方的指数移动平均（二阶矩，自适应学习率）
    3. 偏差修正：修复初始阶段 m, v 偏向 0 的问题
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay  # L2 正则化
        
        self.t = 0  # 时间步
        
        # 初始化一阶和二阶矩
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # L2 正则化（注意：这是 Adam 的方式，不是 AdamW）
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param
                
                # ========== Step 1: 更新一阶矩（梯度均值）==========
                # m = β1 * m + (1 - β1) * grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # ========== Step 2: 更新二阶矩（梯度方差）==========
                # v = β2 * v + (1 - β2) * grad^2
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # ========== Step 3: 偏差修正 ==========
                # 为什么需要？初始时 m, v 接近 0，估计偏低
                # 例如 t=1 时，m = (1-β1)*g1，期望应该是 g1
                # 除以 (1-β1^t) 修正这个偏差
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # ========== Step 4: 更新参数 ==========
                # θ = θ - lr * m_hat / (sqrt(v_hat) + eps)
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ==================== 验证与 PyTorch 对比 ====================
def test_adam():
    torch.manual_seed(42)
    
    # 创建两份相同的参数
    w1 = torch.randn(3, 3, requires_grad=True)
    w2 = w1.clone().detach().requires_grad_(True)
    
    # 我的 Adam
    my_adam = Adam([w1], lr=0.01)
    
    # PyTorch Adam
    torch_adam = torch.optim.Adam([w2], lr=0.01)
    
    # 训练几步
    for step in range(5):
        # 计算相同的 loss
        loss1 = (w1 ** 2).sum()
        loss2 = (w2 ** 2).sum()
        
        # 反向传播
        my_adam.zero_grad()
        loss1.backward()
        my_adam.step()
        
        torch_adam.zero_grad()
        loss2.backward()
        torch_adam.step()
        
        diff = (w1 - w2).abs().max().item()
        print(f"Step {step}: 参数差异 = {diff:.8f}")
    
    assert diff < 1e-6, "实现有误!"
    print("✅ Adam 测试通过!")


if __name__ == "__main__":
    test_adam()
```

---

## AdamW

### 🎯 核心思想

AdamW 修复了 Adam 中 weight decay 的实现问题：

- **Adam**: `grad = grad + λ * w`，然后用 adaptive lr
- **AdamW**: `w = w - λ * w`，直接从权重中减去

**区别**：AdamW 的 weight decay 不会被 adaptive learning rate 缩放。

### 📝 实现代码

```python
class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay
    
    与 Adam 的区别：weight decay 直接作用于权重，
    而不是加到梯度上后被 adaptive lr 缩放。
    
    这是 LLM 训练的标准优化器！
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # ========== AdamW 的关键：weight decay 独立于 Adam ==========
                # 先应用 weight decay（不经过 Adam 的处理）
                if self.weight_decay != 0:
                    param.mul_(1 - self.lr * self.weight_decay)
                
                # 然后正常的 Adam 更新
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# Adam vs AdamW 对比
def compare_adam_adamw():
    """展示 Adam 和 AdamW 的区别"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║               Adam vs AdamW 区别                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Adam with L2:                                               ║
    ║    grad_new = grad + λ * w                                   ║
    ║    w = w - lr * adaptive(grad_new)                           ║
    ║    问题: L2 惩罚被 adaptive lr 缩放了！                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  AdamW (Decoupled):                                          ║
    ║    w = w - lr * λ * w  (直接 decay)                         ║
    ║    w = w - lr * adaptive(grad)                               ║
    ║    正确: weight decay 独立于 adaptive lr                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  为什么重要？                                                ║
    ║  - 高学习率的参数会获得更大的 adaptive scaling               ║
    ║  - Adam 中这些参数的 L2 惩罚也被放大，导致过度正则化         ║
    ║  - AdamW 确保所有参数受到一致的 weight decay                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
```

---

## Learning Rate Scheduler

### 🎯 核心思想

学习率调度：训练过程中动态调整学习率。

常见策略：
- **Warmup**: 开始时从小学习率逐渐增大
- **Cosine Decay**: 余弦曲线下降
- **Linear Decay**: 线性下降

> 🤔 **Q: 为什么需要 Warmup？不 warmup 会怎样？**
>
> 初始时参数随机，梯度方向不稳定。如果直接用大学习率：
> 1. 参数更新幅度大，可能跳过好的局部最小值
> 2. Adam 的 m 和 v 还没积累好，估计不稳定
> 3. BatchNorm/LayerNorm 的统计量还不准
>
> Warmup 让模型先"走稳"，再"跑快"。

### 📝 实现代码

```python
import math

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing
    
    LLM 训练的标准学习率调度
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        # 获取初始学习率
        self.base_lr = optimizer.lr if hasattr(optimizer, 'lr') else optimizer.param_groups[0]['lr']
        
        self.current_step = 0
    
    def get_lr(self) -> float:
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # Warmup: 线性增加
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            # Cosine: 从 1 下降到 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self.get_lr()
        
        # 更新 optimizer 的学习率
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        return lr


class WarmupLinearScheduler:
    """
    Warmup + Linear Decay
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Linear decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return self.base_lr - (self.base_lr - self.min_lr) * progress
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ==================== 可视化 ====================
def visualize_schedulers():
    """可视化学习率调度"""
    import matplotlib.pyplot as plt
    
    total_steps = 10000
    warmup_steps = 1000
    base_lr = 1e-4
    
    # 模拟
    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': base_lr}]
    
    opt = FakeOptimizer()
    
    schedulers = {
        'Cosine': WarmupCosineScheduler(opt, warmup_steps, total_steps),
        'Linear': WarmupLinearScheduler(opt, warmup_steps, total_steps),
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_lr())
            scheduler.step()
        
        plt.plot(lrs, label=name)
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers')
    plt.legend()
    plt.axvline(x=warmup_steps, color='gray', linestyle='--', label='Warmup End')
    plt.savefig('lr_schedulers.png')
    print("已保存: lr_schedulers.png")


if __name__ == "__main__":
    visualize_schedulers()
```

---

## 面试追问汇总

### 基础问题

| 问题 | 答案 |
|:---|:---|
| Adam 的 β1, β2 通常取什么值 | β1=0.9, β2=0.999 |
| 为什么需要偏差修正 | 初始时 m, v 偏向 0，需要放大 |
| Momentum 的物理意义 | 像小球滚下山坡，有惯性 |

### Adam vs AdamW

| 问题 | 答案 |
|:---|:---|
| 两者区别 | AdamW 的 weight decay 不被 adaptive lr 缩放 |
| LLM 用哪个 | AdamW |
| 为什么 AdamW 更好 | 正则化效果更一致，不受梯度大小影响 |

### 学习率调度

| 问题 | 答案 |
|:---|:---|
| 为什么需要 warmup | 初始参数随机，大学习率会跑偏 |
| 常用调度策略 | Warmup + Cosine Decay |
| warmup 步数怎么选 | 通常 1-5% 的总步数 |

### 代码追问

```python
# Q: Adam 的偏差修正为什么是除以 (1 - β^t)？
"""
A: 推导过程
初始 m_0 = 0
m_1 = β1 * 0 + (1-β1) * g1 = (1-β1) * g1
期望 E[m_1] = (1-β1) * E[g1]
想要的是 E[g1]，所以除以 (1-β1)

更一般地:
m_t = (1-β1) * Σ(β1^(t-i) * g_i)
E[m_t] = (1-β1) * Σ(β1^(t-i)) * E[g] = (1 - β1^t) * E[g]
所以 m_hat = m_t / (1 - β1^t)
"""
```

---

## 🔗 相关题目

- [梯度累积](08-efficient-training.md) - 大 batch 训练技巧
- [混合精度训练](08-efficient-training.md) - FP16 + 损失缩放
