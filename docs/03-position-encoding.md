# 📍 位置编码

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

位置编码让 Transformer 感知序列中 token 的位置信息。RoPE 是现代 LLM 的标配，必须掌握！

---

## 目录

- [方法一览对比](#方法一览对比)
- [Sinusoidal Position Encoding](#sinusoidal-position-encoding)
- [Learnable Position Encoding](#learnable-position-encoding)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [ALiBi (Attention with Linear Biases)](#alibi)
- [位置编码对比](#位置编码对比)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：位置编码的核心区别在于**作用方式**和**作用位置**

| 方法 | 作用方式 | 作用位置 | 核心代码差异 | 使用模型 |
|:---|:---|:---|:---|:---|
| **Sinusoidal** | 加法 | Embedding | `x = x + pe` | Transformer |
| **Learnable** | 加法 | Embedding | `x = x + embed(pos)` | BERT, GPT-2 |
| **RoPE** | 乘法(旋转) | Q, K | `q = rotate(q, pos)` | LLaMA, Mistral |
| **ALiBi** | 加法偏置 | Attention Score | `score = score + bias` | BLOOM |

```python
# 四种位置编码的核心区别

# Sinusoidal/Learnable: 加到 embedding 上，影响 Q/K/V
x = x + position_encoding
q, k, v = W_q(x), W_k(x), W_v(x)  # 都含有位置信息

# RoPE: 只旋转 Q 和 K，不影响 V
q, k, v = W_q(x), W_k(x), W_v(x)
q, k = apply_rope(q, pos), apply_rope(k, pos)  # 只影响 Q/K

# ALiBi: 直接加到 attention score 上
score = q @ k.T / sqrt(d) + alibi_bias[distance]  # 不修改 Q/K/V
```

> 🤔 **Q: 为什么 RoPE 只作用于 Q 和 K，不作用于 V？**
>
> Attention 的位置关联性由 Q·K 决定。V 只是被加权的内容，不需要位置信息。
>
> 而且 RoPE 的数学性质（点积只依赖相对位置）只在 Q·K 时成立，应用到 V 没有这个效果。

---

## Sinusoidal Position Encoding

### 🎯 核心思想

原始 Transformer 的位置编码：使用不同频率的 sin/cos 函数，让模型学习相对位置关系。

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**直觉**：
- 不同维度使用不同频率
- 低维度频率高（变化快，捕捉局部位置）
- 高维度频率低（变化慢，捕捉全局位置）

### 📝 实现代码

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal Position Encoding
    
    原始 Transformer 论文 "Attention Is All You Need" 的位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 频率分母: 10000^(2i/d_model)
        # 使用 log 空间计算，避免大数溢出
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度: cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与训练，但会保存到 state_dict）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + position_encoding
        """
        seq_len = x.size(1)
        # 加上位置编码
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def visualize_sinusoidal_pe(d_model: int = 128, max_len: int = 100):
    """可视化位置编码"""
    import matplotlib.pyplot as plt
    
    pe = SinusoidalPositionEncoding(d_model, max_len)
    encoding = pe.pe[0].numpy()  # [max_len, d_model]
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(encoding, cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title('Sinusoidal Position Encoding')
    plt.savefig('sinusoidal_pe.png')
    print("已保存可视化图片: sinusoidal_pe.png")


if __name__ == "__main__":
    # 测试
    pe = SinusoidalPositionEncoding(d_model=512)
    x = torch.randn(2, 100, 512)
    output = pe(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
```

### 💡 面试追问

**Q: 为什么用 sin/cos 而不是其他函数？**

> sin/cos 有一个重要性质：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。这让模型更容易学习相对位置关系。

**Q: 为什么是 10000？**

> 这是一个经验值。它决定了频率范围：最高频率 $1$，最低频率 $1/10000$。

---

## Learnable Position Encoding

### 🎯 核心思想

直接学习每个位置的 embedding，简单粗暴但有效。BERT、GPT-2 等使用。

### 📝 实现代码

```python
class LearnablePositionEncoding(nn.Module):
    """
    可学习位置编码
    
    BERT, GPT-2 等使用
    简单直接，但无法外推到训练时未见过的位置
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 位置 embedding 表
        self.pe = nn.Embedding(max_len, d_model)
        
        # 初始化
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        # [seq_len] -> [seq_len, d_model] -> [1, seq_len, d_model]
        position_embeddings = self.pe(positions).unsqueeze(0)
        
        return self.dropout(x + position_embeddings)
```

**局限性**：无法处理超过 `max_len` 的序列。

---

## Rotary Position Embedding (RoPE)

### 🎯 核心思想

RoPE 将位置信息编码为**旋转**，通过复数乘法实现。

**核心公式**：
$$q_m \cdot k_n = \text{Re}[(R_m q)(R_n k)^*] = f(q, k, m-n)$$

即：经过 RoPE 变换后，Q 和 K 的点积只依赖于**相对位置** $m-n$。

### 📝 实现代码

```python
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 标配
    
    核心思想：
    1. 将 Q, K 向量分成两两一组
    2. 对每组应用 2D 旋转矩阵
    3. 不同维度使用不同的旋转频率
    """
    
    def __init__(
        self,
        dim: int,           # head_dim
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算 cos/sin 缓存"""
        # 位置索引 [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 计算角度 [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 注册缓存 [seq_len, dim/2]
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 到 Q 和 K
        
        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_heads, seq_len, head_dim]
            seq_len: 序列长度
        
        Returns:
            旋转后的 q, k
        """
        if seq_len is None:
            seq_len = q.size(2)
        
        # 获取 cos/sin [seq_len, dim/2]
        cos = self.cos_cached[:seq_len]  # [seq_len, dim/2]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim/2]
        
        # 应用旋转
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        应用旋转变换
        
        旋转公式 (对于每对相邻维度):
        [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
        """
        # 将 x 分成两半: [x0, x1, x2, x3, ...] -> 前半和后半
        # 分成 pairs: (x0, x1), (x2, x3), ... 其中 x1=x[..., :d//2], x2=x[..., d//2:]
        d = x.shape[-1]
        x1 = x[..., :d//2]   # 前一半 [batch, heads, seq, dim/2]
        x2 = x[..., d//2:]   # 后一半 [batch, heads, seq, dim/2]
        
        # 调整 cos/sin 维度以广播 [1, 1, seq, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 应用旋转矩阵
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 拼接回去 [batch, heads, seq, dim]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class RoPESimple(nn.Module):
    """
    RoPE 简化版（面试手写推荐）
    
    使用复数形式，代码更简洁
    """
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Args:
            q, k: [batch, heads, seq_len, head_dim]
        """
        seq_len = q.size(2)
        
        # 计算角度 [seq_len, dim/2]
        t = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # 构造复数形式的旋转因子 e^(i*θ)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # cos + i*sin
        
        # 将 q, k 转为复数 (每两个相邻维度组成一个复数)
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        
        # 复数乘法 = 旋转
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
        q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
        k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
        
        return q_rotated.type_as(q), k_rotated.type_as(k)


# ==================== 验证代码 ====================
def test_rope():
    batch, heads, seq, dim = 2, 8, 64, 64
    
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    
    rope = RotaryPositionEmbedding(dim)
    q_rot, k_rot = rope(q, k)
    
    print(f"Q 形状: {q.shape} -> {q_rot.shape}")
    print(f"K 形状: {k.shape} -> {k_rot.shape}")
    
    # 验证相对位置性质
    # Q_m · K_n 应该只依赖于 m-n
    # 这里简单验证形状正确
    assert q_rot.shape == q.shape
    print("✅ RoPE 测试通过!")


if __name__ == "__main__":
    test_rope()
```

### 🔍 RoPE 的数学原理

> 🤔 **Q: 为什么用旋转而不是加法编码位置？**
>
> 旋转的妙处在于：旋转矩阵的乘法等于角度相加！
>
> $R_m^T \cdot R_n = R_{n-m}$
>
> 所以 $q_m \cdot k_n = (R_m q)^T (R_n k) = q^T R_{n-m} k$，只依赖相对位置！

```
假设二维情况（一对相邻维度）：
原始: q = [q1, q2], k = [k1, k2]

位置 m 的旋转矩阵:
R_m = [[cos(mθ), -sin(mθ)],
       [sin(mθ),  cos(mθ)]]

应用旋转:
q_m = R_m @ q = [q1*cos(mθ) - q2*sin(mθ), q1*sin(mθ) + q2*cos(mθ)]

关键性质（面试必考！）:
q_m · k_n = (R_m @ q)^T @ (R_n @ k)
         = q^T @ R_m^T @ R_n @ k
         = q^T @ R_{n-m} @ k    (因为旋转矩阵相乘 = 角度相加)
         
结论: 点积只依赖于相对位置 (n-m)，而非绝对位置！
```

### 💡 面试追问

**Q: RoPE 如何实现长度外推？**

> 基础 RoPE 外推能力有限。改进方案：
> - **NTK-aware**: 调整 base 参数
> - **YaRN**: 结合 NTK 和插值
> - **Dynamic NTK**: 动态调整频率

**Q: 为什么 RoPE 要在每个 head 内部应用，而不是整个 hidden？**

> 每个 head 独立处理位置信息，让不同 head 可以学习不同的位置关系模式。

> 🤔 **Q: 为什么用 `einsum('i,j->ij', t, inv_freq)` 而不是直接乘？**
>
> `einsum('i,j->ij', t, inv_freq)` 等价于 `torch.outer(t, inv_freq)`，
> 是计算外积：`t[i] * inv_freq[j]` 得到 `[seq_len, dim/2]`。
>
> 直接乘法 `t * inv_freq` 要求维度相同，而这里维度不同。

---

## ALiBi

### 🎯 核心思想

ALiBi (Attention with Linear Biases) 完全不修改 Q/K，而是直接在 attention score 上加一个与距离成正比的偏置。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{m} \cdot \text{dist}\right)V$$

其中 `dist[i,j] = -|i-j|`（距离越远，偏置越负）

> 🤔 **Q: ALiBi 的斜率 m 是怎么设计的？为什么不同 head 斜率不同？**
>
> 斜率 $m = 2^{-8/n}, 2^{-8 \cdot 2/n}, ...$，呈指数递减。
>
> 斜率大的 head 更关注局部（远距离惩罚大），斜率小的 head 能看到更远的信息。
>
> 这让不同 head “分工”关注不同范围。

### 📝 实现代码

```python
class ALiBi(nn.Module):
    """
    Attention with Linear Biases
    
    BLOOM 模型使用
    优点：天然支持长度外推
    """
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        # 每个 head 有不同的斜率 m
        # m = 2^(-8/n), 2^(-8*2/n), ..., 2^(-8*n/n)
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, n: int) -> torch.Tensor:
        """
        计算每个 head 的斜率
        
        使用指数递减的斜率，让不同 head 关注不同范围
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            # 非 2 的幂次，插值
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # 添加额外的斜率
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            slopes = slopes + extra_slopes[:n - closest_power_of_2]
            return torch.tensor(slopes)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成 ALiBi 偏置矩阵
        
        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # 距离矩阵: dist[i,j] = j - i（右边为正，左边为负）
        positions = torch.arange(seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        
        # 因果mask: 只保留下三角（未来位置为负无穷）
        # 这里 distance 保持原样，由调用者决定如何处理因果性
        
        # 乘以斜率: [num_heads, 1, 1] * [seq, seq] -> [num_heads, seq, seq]
        bias = self.slopes.view(-1, 1, 1) * distance.unsqueeze(0)
        
        return bias


def apply_alibi_to_attention(scores: torch.Tensor, alibi_bias: torch.Tensor) -> torch.Tensor:
    """
    将 ALiBi 偏置应用到 attention scores
    
    Args:
        scores: [batch, num_heads, seq_q, seq_k]
        alibi_bias: [num_heads, seq_q, seq_k]
    """
    return scores + alibi_bias.unsqueeze(0)


# ==================== 使用示例 ====================
def test_alibi():
    num_heads = 8
    seq_len = 16
    
    alibi = ALiBi(num_heads)
    bias = alibi(seq_len)
    
    print(f"ALiBi 偏置形状: {bias.shape}")
    print(f"斜率: {alibi.slopes}")
    
    # 可视化第一个 head 的偏置
    print(f"\nHead 0 的偏置矩阵 (前 8x8):")
    print(bias[0, :8, :8].numpy().round(2))


if __name__ == "__main__":
    test_alibi()
```

---

## 位置编码对比

| 方法 | 类型 | 长度外推 | 相对位置 | 计算开销 | 代表模型 |
|:---|:---:|:---:|:---:|:---:|:---|
| Sinusoidal | 加性 | 一般 | 间接 | 低 | Transformer |
| Learnable | 加性 | 差 | 否 | 低 | BERT, GPT-2 |
| RoPE | 乘性 | 一般* | 是 | 中 | LLaMA, Mistral |
| ALiBi | 加性偏置 | 好 | 是 | 低 | BLOOM |

*RoPE 需要额外技术（NTK、YaRN）才能良好外推

---

## 面试追问汇总

### 基础问题

| 问题 | 答案 |
|:---|:---|
| 为什么 Transformer 需要位置编码 | Self-attention 是排列不变的，不含位置信息 |
| Sinusoidal PE 能外推吗 | 理论上可以，但实际效果一般 |
| Learnable PE 的缺点 | 无法处理训练时未见过的长度 |

### RoPE 相关

| 问题 | 答案 |
|:---|:---|
| RoPE 的核心优势 | 自然编码相对位置，Q·K 只依赖于相对距离 |
| RoPE 如何长度外推 | NTK-aware 调整 base，YaRN 结合插值 |
| RoPE 应用在哪里 | 只应用在 Q 和 K，不应用在 V |

### 深度问题

```python
# Q: RoPE 和 Sinusoidal PE 有什么联系？
"""
A: 都使用 sin/cos，但作用方式不同：
- Sinusoidal: 加到 embedding 上，影响 Q, K, V
- RoPE: 乘到 Q, K 上（旋转），不影响 V

RoPE 可以看作是将 Sinusoidal 的想法应用到注意力内部。
"""

# Q: 为什么 LLaMA 选择 RoPE 而不是 ALiBi？
"""
A: 
1. RoPE 在相同长度下效果更好
2. RoPE 通过 NTK-aware 等技术可以外推
3. ALiBi 在非常长的序列上表现更稳定，但短序列可能不如 RoPE
"""
```

---

## 🔗 相关题目

- [Multi-Head Attention](01-attention.md#multi-head-attention) - RoPE 在 MHA 中的应用
- [KV Cache](09-inference-optimization.md#kv-cache) - 位置编码与 KV Cache 的交互
