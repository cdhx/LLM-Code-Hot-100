# 🧠 Attention 机制

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

Attention 是 Transformer 的核心，也是 LLM 面试的重中之重。本章涵盖从基础的 Scaled Dot-Product Attention 到工业级优化的 Flash Attention。

---

## 目录

- [方法一览对比](#方法一览对比)
- [Scaled Dot-Product Attention](#scaled-dot-product-attention)
- [Multi-Head Attention (MHA)](#multi-head-attention)
- [Causal Mask](#causal-mask)
- [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
- [Multi-Query Attention (MQA)](#multi-query-attention-mqa)
- [Flash Attention 原理](#flash-attention-原理)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：MHA/GQA/MQA 的区别就在于 **KV 头数不同**，代码上只差一个参数！

| 方法 | KV 头数 | KV Cache 占用 | 代码差异 | 使用模型 |
|:---|:---:|:---:|:---|:---|
| **MHA** | = Q 头数 | 100% | `num_kv_heads = num_heads` | GPT-3, BERT |
| **GQA** | 介于两者之间 | 12.5%-50% | `num_kv_heads = num_heads // 8` | LLaMA-2 70B, Mistral |
| **MQA** | 1 | 最小(1/h) | `num_kv_heads = 1` | PaLM, Falcon |

```python
# 三种 Attention 的核心区别，就这几行！
class MHA:   W_k = Linear(d, num_heads * head_dim)     # 每个Q头有独立KV
class GQA:   W_k = Linear(d, num_kv_heads * head_dim)  # 多个Q头共享一组KV
class MQA:   W_k = Linear(d, head_dim)                 # 所有Q头共享一个KV
```

> 🤔 **Q: 为什么减少 KV 头数能省内存？**
>
> KV Cache 大小 = `2 × batch × num_kv_heads × seq × head_dim × dtype_bytes`
>
> 减少 `num_kv_heads` 直接减少内存！LLaMA-2 70B 用 GQA (8 KV heads vs 64 Q heads)，省了 87.5%！

---

## Scaled Dot-Product Attention

### 🎯 核心思想

Attention 的本质是**加权求和**：根据 Query 和 Key 的相似度，对 Value 进行加权聚合。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention
    
    Args:
        query: [batch, num_heads, seq_len_q, head_dim]
        key:   [batch, num_heads, seq_len_k, head_dim]
        value: [batch, num_heads, seq_len_v, head_dim]  (seq_len_v == seq_len_k)
        mask:  [batch, 1, seq_len_q, seq_len_k] 或可广播的形状
        dropout_p: Dropout 概率
        training: 是否训练模式
    
    Returns:
        output: [batch, num_heads, seq_len_q, head_dim]
        attn_weights: [batch, num_heads, seq_len_q, seq_len_k]
    """
    d_k = query.size(-1)
    
    # Step 1: 计算注意力分数 Q @ K^T
    # [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
    # -> [batch, heads, seq_q, seq_k]
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: 缩放（防止点积过大导致 softmax 梯度消失）
    # 为什么除以 sqrt(d_k)？
    # 假设 q, k 是均值0方差1的独立随机变量，q·k 的方差 = d_k
    # 除以 sqrt(d_k) 使方差归一化为 1
    scores = scores / math.sqrt(d_k)
    
    # Step 3: 应用 mask（如果有）
    # mask=True/1 的位置会被置为 -inf，softmax 后变成 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Dropout（训练时）
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Step 6: 加权求和
    # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
    # -> [batch, heads, seq_q, head_dim]
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights
```

### 🔍 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|:---|:---:|:---:|
| Q @ K^T | O(n² · d) | O(n²) |
| Softmax | O(n²) | O(n²) |
| Attn @ V | O(n² · d) | O(n · d) |
| **总计** | **O(n² · d)** | **O(n²)** |

> ⚠️ **瓶颈**：n² 的复杂度是长序列的主要瓶颈！

---

## Multi-Head Attention

### 🎯 核心思想

多头注意力让模型在**不同的表示子空间**中学习不同的注意力模式，然后合并结果。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 完整实现
    
    MHA(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
    其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
    """
    
    def __init__(
        self, 
        d_model: int,           # 模型维度（如 768, 1024）
        num_heads: int,         # 注意力头数（如 12, 16）
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # 检查维度能否整除
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 线性投影层
        # 实际实现中通常合并 Q, K, V 的投影以提高效率
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier 初始化"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key:   [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model]
            mask:  [batch, seq_len_q, seq_len_k] 或可广播
            need_weights: 是否返回注意力权重
        
        Returns:
            output: [batch, seq_len_q, d_model]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_k] (如果 need_weights=True)
        """
        batch_size = query.size(0)
        
        # ========== Step 1: 线性投影 ==========
        # [batch, seq, d_model] -> [batch, seq, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # ========== Step 2: 拆分多头 ==========
        # [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
        # -> [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ========== Step 3: 计算注意力 ==========
        # 调整 mask 维度 [batch, 1, 1, seq_k] 用于广播
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
        # -> [batch, heads, seq_q, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # ========== Step 4: 合并多头 ==========
        # [batch, heads, seq_q, head_dim] -> [batch, seq_q, heads, head_dim]
        # -> [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # ========== Step 5: 输出投影 ==========
        output = self.W_o(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None


# ==================== 验证代码 ====================
def test_mha():
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 自注意力
    output, weights = mha(x, x, x, need_weights=True)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    assert output.shape == x.shape, "输出维度错误!"
    print("✅ MHA 测试通过!")

if __name__ == "__main__":
    test_mha()
```

### 💡 面试追问

**Q1: 为什么要用多头而不是一个大的 Attention？**

> 多头让模型在不同子空间学习不同的注意力模式（如语法、语义、位置关系），类似 CNN 中的多个 filter。

**Q2: head_dim 和 num_heads 怎么选？**

> 通常 d_model / num_heads = 64 或 128。头数太多单头容量不足，太少多样性不够。

**Q3: 为什么要除以 √d_k？**

> 假设 q, k 是独立的均值 0 方差 1 的随机变量，点积的方差 = d_k。不缩放的话，d_k 大时 softmax 输入值大，梯度趋近于 0。

> 🤔 **Q: transpose(1, 2) 是在干什么？为什么需要 contiguous()？**
>
> `transpose(1, 2)` 把 `[batch, seq, heads, dim]` 变成 `[batch, heads, seq, dim]`，让每个 head 可以独立做矩阵乘法。
>
> `contiguous()` 是因为 transpose 只改变 stride 不改变内存布局，view() 需要连续内存。

---

## Causal Mask

### 🎯 核心思想

因果掩码确保每个位置**只能看到自己和之前的位置**，这是自回归生成的基础。

### 📝 实现代码

```python
def create_causal_mask(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    创建因果掩码（下三角矩阵）
    
    返回的 mask 中:
    - 1 表示可以看到（保留）
    - 0 表示不能看到（mask 掉）
    
    示例 (seq_len=4):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    # 方法1: 使用 torch.tril（推荐）
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    
    return mask


def create_causal_mask_additive(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    创建加性因果掩码（直接加到 attention scores 上）
    
    返回的 mask 中:
    - 0 表示可以看到
    - -inf 表示不能看到
    
    这种形式可以直接和 scores 相加: scores = scores + mask
    """
    # 上三角为 -inf，其他为 0
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1  # 从对角线上方一格开始
    ).to(dtype)
    
    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    滑动窗口掩码（Longformer/Mistral 使用）
    
    每个位置只能看到前 window_size 个位置
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1
    
    return mask


# ==================== 可视化 ====================
def visualize_mask(mask: torch.Tensor, title: str = "Attention Mask"):
    """打印 mask 矩阵"""
    print(f"\n{title}:")
    print(mask.int().numpy())


if __name__ == "__main__":
    # 因果掩码
    causal = create_causal_mask(5)
    visualize_mask(causal, "Causal Mask")
    
    # 滑动窗口掩码
    sliding = create_sliding_window_mask(5, window_size=3)
    visualize_mask(sliding, "Sliding Window Mask (window=3)")
```

输出示例:
```
Causal Mask:
[[1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [1 1 1 1 0]
 [1 1 1 1 1]]

Sliding Window Mask (window=3):
[[1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [0 1 1 1 0]
 [0 0 1 1 1]]
```

> 🤔 **Q: mask 为什么用 0 和 1，不直接用 -inf？**
>
> 两种风格都可以！上面代码用 0/1 是布尔掩码风格（`mask==0` 的位置填 -inf）。
>
> 也可以用加性掩码：`scores = scores + mask`，其中 mask 直接是 0 或 -inf。两种等价，看团队习惯。

---

## Grouped-Query Attention (GQA)

### 🎯 核心思想

GQA 是 MHA 和 MQA 的折中：**多个 Query 头共享同一组 Key-Value 头**。

- MHA: num_kv_heads = num_heads（每个 Q 头有独立的 KV）
- MQA: num_kv_heads = 1（所有 Q 头共享一个 KV）
- GQA: 1 < num_kv_heads < num_heads（分组共享）

**优势**：减少 KV Cache 内存，同时保持接近 MHA 的效果。

### 📝 实现代码

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    
    LLaMA-2 70B, Mistral 等使用
    通过减少 KV 头数来降低 KV Cache 内存
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,          # Query 头数
        num_kv_heads: int = None,  # KV 头数（默认等于 num_heads，即 MHA）
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        
        # 检查 num_heads 能否被 num_kv_heads 整除
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        
        # 每个 KV 头对应的 Q 头数
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Q 投影: d_model -> num_heads * head_dim
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # K, V 投影: d_model -> num_kv_heads * head_dim（减少了！）
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 投影
        Q = self.W_q(x)  # [batch, seq, num_heads * head_dim]
        K = self.W_k(x)  # [batch, seq, num_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq, num_kv_heads * head_dim]
        
        # 重塑 Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 重塑 K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # ========== 关键：扩展 KV 以匹配 Q 头数 ==========
        # 每个 KV 头重复 num_queries_per_kv 次
        # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # 标准 Attention 计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        
        return output


# ==================== 内存分析 ====================
def analyze_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    dtype_bytes: int = 2  # FP16 = 2 bytes
):
    """分析 KV Cache 内存占用"""
    head_dim = d_model // num_heads
    
    # MHA: 2 * batch * num_heads * seq * head_dim
    mha_memory = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    
    # GQA: 2 * batch * num_kv_heads * seq * head_dim
    gqa_memory = 2 * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
    
    print(f"=== KV Cache 内存分析 ===")
    print(f"配置: batch={batch_size}, seq={seq_len}, d_model={d_model}")
    print(f"       num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"MHA 内存: {mha_memory / 1024**2:.2f} MB")
    print(f"GQA 内存: {gqa_memory / 1024**2:.2f} MB")
    print(f"节省: {(1 - gqa_memory/mha_memory) * 100:.1f}%")


if __name__ == "__main__":
    # 模拟 LLaMA-2 70B 配置
    analyze_kv_cache_memory(
        batch_size=1,
        seq_len=4096,
        d_model=8192,
        num_heads=64,
        num_kv_heads=8  # GQA 配置
    )
```

### 💡 面试追问

**Q: GQA 和 MQA 的区别？**

| 方法 | num_kv_heads | KV Cache 内存 | 质量 |
|:---|:---:|:---:|:---:|
| MHA | = num_heads | 100% | 最好 |
| GQA | 介于两者之间 | 12.5%-50% | 接近 MHA |
| MQA | 1 | 最小 | 略有下降 |

---

## Multi-Query Attention (MQA)

### 🎯 核心思想

MQA 是 GQA 的极端情况：**所有 Query 头共享同一个 Key-Value**。

> 🤔 **Q: repeat_interleave 和 expand 有什么区别？能不能用 expand？**
>
> 可以！`expand` 不复制内存，`repeat_interleave` 会复制。
>
> 实际上用 `expand` 更高效：`K = K.expand(-1, self.num_heads, -1, -1)`
>
> 但 `repeat_interleave` 语义更清晰，面试写这个没问题。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    
    所有 Query 头共享同一组 K, V
    极大减少 KV Cache，但可能损失一些表达能力
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q: 多头
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K, V: 只有一个头！
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, seq, num_heads * head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        
        # K, V: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)
        
        # 广播：K, V 自动扩展到 [batch, num_heads, seq, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(output)
```

---

## Flash Attention 原理

### 🎯 核心思想

Flash Attention 通过**分块计算 + Online Softmax**，避免存储完整的 N×N Attention 矩阵，大幅减少内存访问。

**关键 insight**：GPU 的瓶颈往往是内存带宽而非计算能力。

### 📝 伪代码实现

```python
def flash_attention_forward_naive(Q, K, V, block_size=64):
    """
    Flash Attention 前向传播（简化版伪代码）
    
    核心思想：
    1. 分块处理，避免存储完整的 N×N attention 矩阵
    2. 使用 Online Softmax 在块间增量更新
    
    标准 Attention 内存：O(N²) - 存储完整 attention 矩阵
    Flash Attention 内存：O(N) - 只存储输出和中间统计量
    """
    N, d = Q.shape
    
    # 输出和统计量初始化
    O = torch.zeros_like(Q)           # 输出
    L = torch.zeros(N, 1)             # logsumexp (softmax 分母的 log)
    M = torch.full((N, 1), float('-inf'))  # running max (数值稳定性)
    
    # 分块遍历 K, V
    for j in range(0, N, block_size):
        j_end = min(j + block_size, N)
        Kj = K[j:j_end]  # 当前 K 块
        Vj = V[j:j_end]  # 当前 V 块
        
        # 分块遍历 Q
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            Qi = Q[i:i_end]  # 当前 Q 块
            
            # === 在 SRAM 中计算（不写回 HBM）===
            
            # 1. 计算局部 attention scores
            Sij = Qi @ Kj.T  # [block_size, block_size]
            
            # 2. Online Softmax 更新
            # 找到新的 max
            Mij = Sij.max(dim=-1, keepdim=True).values
            M_new = torch.max(M[i:i_end], Mij)
            
            # 计算缩放因子
            exp_old = torch.exp(M[i:i_end] - M_new)  # 旧结果的缩放
            exp_new = torch.exp(Sij - M_new)         # 新块的 exp
            
            # 更新 logsumexp
            L_new = exp_old * L[i:i_end] + exp_new.sum(dim=-1, keepdim=True)
            
            # 3. 更新输出（增量更新）
            # 关键公式：旧输出要乘以旧的归一化因子 L_old，再除以新的 L_new
            O[i:i_end] = (exp_old * L[i:i_end] * O[i:i_end] + exp_new @ Vj) / L_new
            
            # 更新统计量
            M[i:i_end] = M_new
            L[i:i_end] = L_new
    
    return O


def explain_flash_attention_io():
    """Flash Attention IO 复杂度分析"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Flash Attention IO 分析                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  标准 Attention HBM 访问:                                    ║
    ║  1. 读 Q, K, V:        3 × N × d                            ║
    ║  2. 写 S = QK^T:       N × N        ← 瓶颈!                 ║
    ║  3. 读 S 做 softmax:   N × N                                ║
    ║  4. 写 P = softmax(S): N × N                                ║
    ║  5. 读 P, V:           2 × N × d                            ║
    ║  总计: O(N²) HBM 访问                                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Flash Attention HBM 访问:                                   ║
    ║  1. 读 Q, K, V: 3 × N × d                                   ║
    ║  2. 分块计算，中间结果 S, P 在 SRAM，不写 HBM                ║
    ║  3. 写输出 O:  N × d                                        ║
    ║  总计: O(N × d) HBM 访问                                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  加速比: 约 2-4x (取决于序列长度和 GPU)                      ║
    ║  内存节省: O(N²) → O(N)                                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
```

### 💡 面试追问

**Q: Flash Attention 为什么能加速？**

> 1. 减少 HBM 访问（GPU 内存带宽是瓶颈）
> 2. 分块计算充分利用 SRAM（L1/L2 cache，速度快 10-100 倍）
> 3. 不存储 N×N 的 attention 矩阵

**Q: Online Softmax 是什么？**

> 增量计算 softmax：维护 running max 和 running sum，每处理一个块就更新，无需等到看完所有数据。

---

## 面试追问汇总

### 计算复杂度

| 问题 | 答案 |
|:---|:---|
| MHA 时间复杂度 | O(n² · d)，n 是序列长度，d 是模型维度 |
| MHA 空间复杂度 | O(n²)（存储 attention 矩阵） |
| 为什么长序列慢 | n² 复杂度，4096 tokens → 1600万次计算 |

### KV Cache

| 问题 | 答案 |
|:---|:---|
| KV Cache 作用 | 缓存历史 K, V，避免重复计算 |
| 内存计算公式 | 2 × batch × num_kv_heads × seq × head_dim × dtype_bytes |
| GQA 省多少内存 | 如果 num_kv_heads = num_heads/8，省 87.5% |

### 数值稳定性

| 问题 | 答案 |
|:---|:---|
| 为什么要缩放 | 点积方差随维度线性增长，不缩放会导致 softmax 梯度消失 |
| 为什么减 max | 防止 exp 溢出，softmax(x) = softmax(x - max(x)) |
| FP16 注意事项 | Softmax 在 FP32 计算，其他可以 FP16 |

---

## 🔗 相关题目

- [RoPE 位置编码](03-position-encoding.md#rope) - 现代 LLM 标配
- [KV Cache](09-inference-optimization.md#kv-cache) - 推理必备
- [Transformer 架构](10-transformer-architecture.md) - 完整 Encoder/Decoder
