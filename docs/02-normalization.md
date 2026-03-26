# 📏 归一化层

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐

归一化是深度学习训练稳定的关键。LLM 领域主要使用 Layer Normalization 和 RMSNorm。

---

## 目录

- [方法一览对比](#方法一览对比)
- [Layer Normalization](#layer-normalization)
- [RMS Normalization](#rms-normalization)
- [Batch Normalization](#batch-normalization)
- [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：归一化的核心区别在于**在哪个维度计算统计量**

| 方法 | 统计维度 | 公式差异 | 使用场景 |
|:---|:---|:---|:---|
| **LayerNorm** | 每个样本的特征维度 | `(x - mean) / std * γ + β` | Transformer |
| **RMSNorm** | 每个样本的特征维度 | `x / RMS(x) * γ`（无 mean） | LLaMA, Mistral |
| **BatchNorm** | 整个 batch 的同一特征 | 同 LayerNorm，但维度不同 | CNN |

```python
# 三种 Norm 的核心区别，就这几行！
# 输入 x: [batch, seq, hidden]

# LayerNorm: 对每个 token 的 hidden 维度归一化
mean = x.mean(dim=-1)          # [batch, seq]

# RMSNorm: 同样对 hidden 维度，但只用 RMS，不用 mean
rms = sqrt(mean(x**2, dim=-1)) # [batch, seq]

# BatchNorm: 对每个特征在 batch 维度归一化
mean = x.mean(dim=0)           # [seq, hidden]
```

> 🤔 **Q: 为什么 LLM 不用 BatchNorm？**
>
> 1. 序列长度变化：不同样本长度不同，BatchNorm 难以处理
> 2. 小 batch 不稳定：LLM 训练通常用小 batch，统计量方差大
> 3. 推理时不一致：训练用 batch 统计量，推理用 running stats，有差异

---

## Layer Normalization

### 🎯 核心思想

对**每个样本的特征维度**进行归一化，使其均值为 0，方差为 1，然后通过可学习参数进行缩放和平移。

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$（特征维度的均值）
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$（特征维度的方差）
- $\gamma, \beta$ 是可学习参数

### 📝 实现代码

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization 手动实现
    
    对最后 normalized_shape 维度进行归一化
    例如: 输入 [batch, seq, hidden]，normalized_shape=hidden
          则对每个 [hidden] 向量归一化
    """
    
    def __init__(
        self, 
        normalized_shape: int | tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True  # 是否使用可学习参数
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # 可学习的缩放和平移参数
            self.weight = nn.Parameter(torch.ones(normalized_shape))   # gamma
            self.bias = nn.Parameter(torch.zeros(normalized_shape))    # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., *normalized_shape]
        
        Returns:
            归一化后的张量，形状不变
        """
        # 确定归一化的维度
        # 如果 normalized_shape = (hidden,)，则 dims = (-1,)
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Step 1: 计算均值
        mean = x.mean(dim=dims, keepdim=True)
        
        # Step 2: 计算方差（使用无偏估计 unbiased=False 与 PyTorch 保持一致）
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Step 3: 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Step 4: 缩放和平移
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        return x_normalized


# ==================== 验证代码 ====================
def test_layer_norm():
    batch, seq, hidden = 2, 10, 768
    
    # 我们的实现
    my_ln = LayerNorm(hidden)
    # PyTorch 官方实现
    torch_ln = nn.LayerNorm(hidden)
    
    # 同步参数
    torch_ln.weight.data = my_ln.weight.data.clone()
    torch_ln.bias.data = my_ln.bias.data.clone()
    
    x = torch.randn(batch, seq, hidden)
    
    my_output = my_ln(x)
    torch_output = torch_ln(x)
    
    diff = (my_output - torch_output).abs().max()
    print(f"最大误差: {diff:.8f}")
    assert diff < 1e-5, "实现有误!"
    print("✅ LayerNorm 测试通过!")


if __name__ == "__main__":
    test_layer_norm()
```

### 🔍 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|:---|:---:|:---:|
| 计算均值 | O(d) | O(1) |
| 计算方差 | O(d) | O(1) |
| 归一化 | O(d) | O(d) |
| **总计** | **O(d)** | **O(d)** |

> 🤔 **Q: 为什么 LayerNorm 用 `unbiased=False`？**
>
> PyTorch 的 `var()` 默认用无偏估计（除以 n-1），但 LayerNorm 的原论文用的是有偏估计（除以 n）。
>
> 要与 PyTorch 官方 `nn.LayerNorm` 结果一致，必须用 `unbiased=False`！

---

## RMS Normalization

### 🎯 核心思想

RMSNorm 是 LayerNorm 的简化版本：**去掉均值中心化**，只做方差归一化。

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}$$

其中：
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

**优势**：
- 计算更快（少一次 mean 计算）
- 实验表明在 LLM 中效果与 LayerNorm 相当
- LLaMA、Mistral、DeepSeek 等都使用 RMSNorm

### 📝 实现代码

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 使用
    相比 LayerNorm 去掉了 mean centering，更高效
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数，没有偏移参数
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim]
        
        Returns:
            归一化后的张量
        """
        # 计算 RMS: sqrt(mean(x^2))
        # 等价于 x 的 L2 范数除以 sqrt(dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        return x / rms * self.weight
    
    def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        优化版本：使用 rsqrt 避免两次开方
        """
        # rsqrt = 1 / sqrt(x)，比 sqrt 后再除法更快
        rms_inv = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight


class RMSNormWithCast(nn.Module):
    """
    带类型转换的 RMSNorm（处理混合精度训练）
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        
        # 在 float32 下计算，数值更稳定
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # 转回原始类型
        return (self.weight * x).to(input_dtype)


# ==================== 对比测试 ====================
def compare_norms():
    """对比 LayerNorm 和 RMSNorm"""
    import time
    
    dim = 4096
    batch, seq = 32, 512
    x = torch.randn(batch, seq, dim).cuda()
    
    ln = LayerNorm(dim).cuda()
    rms = RMSNorm(dim).cuda()
    
    # 预热
    for _ in range(10):
        _ = ln(x)
        _ = rms(x)
    
    torch.cuda.synchronize()
    
    # LayerNorm 计时
    start = time.perf_counter()
    for _ in range(100):
        _ = ln(x)
    torch.cuda.synchronize()
    ln_time = time.perf_counter() - start
    
    # RMSNorm 计时
    start = time.perf_counter()
    for _ in range(100):
        _ = rms(x)
    torch.cuda.synchronize()
    rms_time = time.perf_counter() - start
    
    print(f"LayerNorm: {ln_time*1000:.2f} ms")
    print(f"RMSNorm:   {rms_time*1000:.2f} ms")
    print(f"加速比:    {ln_time/rms_time:.2f}x")


if __name__ == "__main__":
    test_layer_norm()
```

### 💡 面试追问

**Q: 为什么 LLM 用 RMSNorm 而不是 LayerNorm？**

> 1. **更快**：省掉 mean 计算，约快 10-15%
> 2. **效果相当**：实验表明在 LLM 场景下，mean centering 贡献不大
> 3. **参数更少**：没有 bias 参数

**Q: RMSNorm 的数学直觉是什么？**

> RMSNorm 可以理解为将向量投影到单位球面上再缩放。它保持向量方向，只调整模长。

---

## Batch Normalization

### 🎯 核心思想

对**每个特征在 batch 维度**进行归一化。主要用于 CNN，在 Transformer 中较少使用。

> 🤔 **Q: 为什么 BatchNorm 训练和推理表现不一致？**
>
> 训练时用当前 batch 的 mean/var，推理时用积累的 running_mean/running_var。
>
> 如果训练数据和推理数据分布不同，会有偏差。这也是 LLM 不用 BatchNorm 的原因之一。

### 📝 实现代码

```python
class BatchNorm1d(nn.Module):
    """
    Batch Normalization 手动实现
    
    对 batch 维度归一化（每个特征通道独立）
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        
        if track_running_stats:
            # running stats 不参与梯度计算
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_features] 或 [batch, num_features, length]
        """
        # 对于 3D 输入，先 flatten 为 2D 计算统计量
        original_shape = x.shape
        if x.dim() == 3:
            # [batch, features, length] -> [batch * length, features]
            x = x.permute(0, 2, 1).reshape(-1, self.num_features)
            reshape_back = True
        else:
            reshape_back = False
        
        if self.training:
            # 训练时：使用当前 batch 的统计量
            # x: [N, features]，对 dim=0 取均值得 [features]
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新 running stats（指数移动平均）
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
        else:
            # 推理时：使用 running stats
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        if reshape_back:
            # [batch * length, features] -> [batch, features, length]
            batch_size, _, length = original_shape
            x_normalized = x_normalized.reshape(batch_size, length, -1).permute(0, 2, 1)
        
        return x_normalized
```

### 💡 BatchNorm vs LayerNorm

| 特性 | BatchNorm | LayerNorm |
|:---|:---|:---|
| 归一化维度 | Batch 维度 | 特征维度 |
| 依赖 batch size | 是（需要足够大） | 否 |
| 训练/推理一致性 | 不同（running stats） | 相同 |
| 适用场景 | CNN | Transformer, RNN |
| 序列长度变化 | 需要特殊处理 | 天然支持 |

---

## Pre-Norm vs Post-Norm

### 🎯 核心区别

> 💡 **记忆技巧**：Pre-Norm “先洗澡再进门”，Post-Norm “进门后再洗澡”

```
Post-Norm (原始 Transformer):
x = LayerNorm(x + Sublayer(x))   # 注意: Norm 在最外层

Pre-Norm (现代 LLM):
x = x + Sublayer(LayerNorm(x))   # 注意: Norm 在最里层，残差在外面
```

### 📝 实现对比

```python
class PostNormBlock(nn.Module):
    """Post-Norm: 原始 Transformer 架构"""
    
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先计算残差，再归一化
        return self.norm(x + self.sublayer(x))


class PreNormBlock(nn.Module):
    """Pre-Norm: 现代 LLM 架构（GPT-2+, LLaMA 等）"""
    
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先归一化，再计算残差
        return x + self.sublayer(self.norm(x))


# Pre-Norm 的梯度流分析
def analyze_gradient_flow():
    """
    Pre-Norm 的优势：梯度可以直接通过残差连接回传
    
    Post-Norm: grad 需要经过 LayerNorm
    ∂L/∂x = ∂L/∂y · ∂LayerNorm/∂(x + sublayer(x))
    
    Pre-Norm: grad 有直接的恒等路径
    ∂L/∂x = ∂L/∂y · (1 + ∂sublayer/∂LayerNorm · ∂LayerNorm/∂x)
    
    第一项的 1 保证了梯度能直接回传，避免梯度消失
    """
    print("""
    Pre-Norm 优势:
    1. 梯度直通路径，训练更稳定
    2. 不需要 warmup 或特殊学习率调度
    3. 可以训练更深的模型
    
    Pre-Norm 的劣势:
    1. 最终输出前需要额外的 LayerNorm
    2. 理论上表达能力略弱（但实践中可忽略）
    """)

> 🤔 **Q: 为什么 Pre-Norm 梯度更稳定？看这个图：**
>
> ```
> Post-Norm: x ──▶ Sublayer ──▶ + ──▶ LayerNorm ──▶ output
>            └─────────────┘
>            梯度必须经过 LayerNorm，可能被压缩
>
> Pre-Norm:  x ──▶ LayerNorm ──▶ Sublayer ──▶ + ──▶ output
>            └────────────────────┘
>            梯度可以通过残差连接直接回传，永远不会消失！
> ```

---

## 面试追问汇总

### 基础问题

| 问题 | 答案 |
|:---|:---|
| LayerNorm 归一化哪个维度 | 最后的特征维度（如 hidden_size） |
| 为什么不用 BatchNorm | 序列长度变化、小 batch 不稳定 |
| eps 的作用 | 防止除零，保证数值稳定 |

### 进阶问题

| 问题 | 答案 |
|:---|:---|
| RMSNorm 比 LayerNorm 快多少 | 约 10-15%（省去 mean 计算） |
| Pre-Norm 的优势 | 梯度直通路径，训练更稳定 |
| 为什么 LLaMA 用 Pre-Norm | 可以训练更深模型，不需要 warmup |

### 代码追问

```python
# Q: 这段代码有什么问题？
def buggy_layer_norm(x, eps=1e-5):
    mean = x.mean()  # 错误！应该指定 dim
    var = x.var()    # 错误！应该指定 dim
    return (x - mean) / (var + eps) ** 0.5  # 也可以工作，但不标准

# A: 应该是
def correct_layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
```

---

## 🔗 相关题目

- [Transformer 架构](10-transformer-architecture.md) - 完整模块组装
- [Attention 机制](01-attention.md) - Attention 前后的 Norm
