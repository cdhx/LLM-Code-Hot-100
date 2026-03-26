# 🏗️ Transformer 架构

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐

完整的 Transformer 架构是所有 LLM 的基础。本章整合所有组件！

---

## 目录

- [架构一览对比](#架构一览对比)
- [Transformer 概览](#transformer-概览)
- [Feed-Forward Network (FFN)](#feed-forward-network)
- [Transformer Encoder Layer](#transformer-encoder-layer)
- [Transformer Decoder Layer](#transformer-decoder-layer)
- [GPT-style Decoder-Only](#gpt-style-decoder-only)
- [残差连接与层归一化](#残差连接与层归一化)
- [完整 Transformer 实现](#完整-transformer-实现)
- [面试追问汇总](#面试追问汇总)

---

## 架构一览对比

> 💡 **一句话区分**：不同架构的核心区别在于 **Attention 是否有掩码** 和 **是否有 Cross Attention**

| 架构 | Self-Attn | Cross-Attn | Mask | 代表模型 |
|:---|:---:|:---:|:---|:---|
| **Encoder-Only** | ✓ | ✗ | 无 | BERT |
| **Decoder-Only** | ✓ | ✗ | Causal | GPT, LLaMA |
| **Encoder-Decoder** | ✓ | ✓ | Causal(Dec) | T5, BART |

```python
# 三种架构的核心区别

# Encoder-Only (BERT): 看得到所有 token，适合理解
output = SelfAttn(x, x, x, mask=None)       # 全部可见

# Decoder-Only (GPT): 只能看到左边，适合生成
output = SelfAttn(x, x, x, mask=causal)     # 只能看左边

# Encoder-Decoder (T5): Encoder 全可见，Decoder 用 Encoder 输出
enc_out = EncoderSelfAttn(x, x, x, mask=None)  # Encoder 全可见
dec_out = DecoderSelfAttn(y, y, y, mask=causal) # Decoder 只看左边
final = CrossAttn(dec_out, enc_out, enc_out)    # 看 Encoder
```

> 🤔 **Q: 为什么现代 LLM 都用 Decoder-Only？**
>
> 1. **简单**：一个统一架构，无需分 Encoder/Decoder
> 2. **扩展性好**：只需堆堆层，没有复杂的 cross-attention
> 3. **训练高效**：每个 position 都是一个训练样本（next token prediction）
> 4. **推理简单**：KV Cache 实现简单，不用处理 Encoder

---

## Transformer 概览

```
原始 Transformer (Encoder-Decoder):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    Encoder                         Decoder              │
│    ┌─────────┐                    ┌─────────┐          │
│    │  MHA    │                    │ Masked  │          │
│    │ (Self)  │                    │  MHA    │          │
│    └────┬────┘                    └────┬────┘          │
│         │                              │               │
│    ┌────┴────┐                    ┌────┴────┐          │
│    │   FFN   │                    │  Cross  │          │
│    └────┬────┘                    │   MHA   │          │
│         │                         └────┬────┘          │
│    x N layers                          │               │
│                                   ┌────┴────┐          │
│                                   │   FFN   │          │
│                                   └────┬────┘          │
│                                        │               │
│                                   x N layers           │
└─────────────────────────────────────────────────────────┘

现代 LLM (Decoder-Only):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    Decoder-Only (GPT/LLaMA/...)                        │
│    ┌─────────┐                                          │
│    │ Masked  │  ← Causal Mask 保证自回归                │
│    │  MHA    │                                          │
│    └────┬────┘                                          │
│         │                                               │
│    ┌────┴────┐                                          │
│    │   FFN   │  ← 通常 4x 或 8/3x hidden_size          │
│    └────┬────┘                                          │
│         │                                               │
│    x N layers                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Feed-Forward Network

### 🎯 核心思想

FFN 是 Transformer 的"记忆库"，存储知识。通常是两层 MLP：

$$\text{FFN}(x) = \text{Act}(xW_1 + b_1)W_2 + b_2$$

**维度变化**：d_model → d_ff → d_model（通常 d_ff = 4 × d_model）

> 🤔 **Q: 为什么 FFN 要先升维再降维？不是浪费计算吗？**
>
> 升维是为了表达能力！线性变换 + 非线性激活在高维空间更容易分离特征。
>
> 这也是为什么研究表明 FFN 是"记忆库"——它存储了大量知识。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    标准 FFN (GPT-2 风格)
    
    d_model -> d_ff -> d_model
    激活函数: GELU
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        x = self.w1(x)           # [batch, seq, d_ff]
        x = self.activation(x)    # 非线性
        x = self.dropout(x)
        x = self.w2(x)           # [batch, seq, d_model]
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU FFN (LLaMA 风格)
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 使用
    
    SwiGLU(x) = (xW1 * SiLU(xW_gate)) @ W2
    
    参数量: 3 * d_model * d_ff（比标准 FFN 多 50%）
    但效果更好，通常用更小的 d_ff 来补偿
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # LLaMA 的 d_ff 计算: 2/3 * 4 * d_model，然后取 256 的倍数
        d_ff = d_ff or int(2 * 4 * d_model / 3)
        d_ff = ((d_ff + 255) // 256) * 256  # 对齐到 256
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # Gate 投影
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W_gate) * (x @ W1) @ W2
        gate = F.silu(self.w_gate(x))  # SiLU = x * sigmoid(x)
        x = self.w1(x) * gate          # 门控
        x = self.dropout(x)
        x = self.w2(x)
        return x


class GeGLU(nn.Module):
    """
    GeGLU FFN (另一种门控变体)
    
    GeGLU(x) = GELU(xW_gate) * (xW1) @ W2
    """
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 合并 W1 和 W_gate 为一个大矩阵
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.gelu(gate) * up
        x = self.w_down(x)
        return x
```

---

## Transformer Encoder Layer

### 📝 实现代码

```python
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (BERT 风格)
    
    Pre-Norm 架构:
    x -> LayerNorm -> Self-Attn -> Add -> LayerNorm -> FFN -> Add
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm"
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norms
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        mask: [seq_len, seq_len] attention mask
        key_padding_mask: [batch, seq_len] padding mask
        """
        # Pre-Norm Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)
        
        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x


class RMSNorm(nn.Module):
    """RMSNorm 实现"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## Transformer Decoder Layer

### 📝 实现代码

```python
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer (原始 Transformer 风格)
    
    包含:
    1. Masked Self-Attention (因果)
    2. Cross-Attention (对 Encoder 输出)
    3. FFN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norms (Pre-Norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [batch, tgt_len, d_model] decoder 输入
        encoder_output: [batch, src_len, d_model] encoder 输出
        """
        # Masked Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=self_attn_mask
        )
        x = residual + self.dropout(attn_out)
        
        # Cross-Attention
        residual = x
        x = self.norm2(x)
        cross_out, _ = self.cross_attn(
            x,                # Query: 来自 decoder
            encoder_output,   # Key: 来自 encoder
            encoder_output,   # Value: 来自 encoder
            attn_mask=cross_attn_mask
        )
        x = residual + self.dropout(cross_out)
        
        # FFN
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x
```

---

## GPT-style Decoder-Only

### 🎯 核心思想

现代 LLM（GPT、LLaMA、Mistral 等）都使用 Decoder-Only 架构：
- 没有 Encoder，没有 Cross-Attention
- 只有 Causal Self-Attention
- 更简单，更容易扩展

### 📝 实现代码

```python
class GPTDecoderLayer(nn.Module):
    """
    GPT/LLaMA 风格的 Decoder Layer
    
    只有 Masked Self-Attention + FFN
    没有 Cross-Attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = None,  # GQA
        d_ff: int = None,
        dropout: float = 0.0,
        use_swiglu: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 使用 GQA 或 MHA
        if num_kv_heads:
            self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        
        # 使用 SwiGLU 或标准 FFN
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout, activation="gelu")
        
        # RMSNorm (现代 LLM 标配)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache = None
    ) -> tuple[torch.Tensor, any]:
        """
        x: [batch, seq_len, d_model]
        mask: causal mask
        kv_cache: KV Cache 对象
        """
        # Pre-Norm Masked Self-Attention
        residual = x
        x = self.norm1(x)
        
        if hasattr(self.self_attn, 'forward_with_cache'):
            # 自定义 Attention（支持 KV Cache）
            attn_out, kv_cache = self.self_attn.forward_with_cache(x, mask, kv_cache)
        else:
            # PyTorch 标准 MHA
            attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        
        x = residual + attn_out
        
        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x, kv_cache


class GroupedQueryAttention(nn.Module):
    """GQA 实现（简化版）"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 扩展 KV
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=2)
        V = V.repeat_interleave(repeat_factor, dim=2)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out), None
```

---

## 残差连接与层归一化

### 🎯 Post-Norm vs Pre-Norm

```python
def post_norm_block(x, sublayer, norm):
    """
    Post-Norm (原始 Transformer)
    
    x -> Sublayer -> Add -> Norm
    
    问题: 深层网络梯度不稳定
    """
    return norm(x + sublayer(x))


def pre_norm_block(x, sublayer, norm):
    """
    Pre-Norm (现代 LLM)
    
    x -> Norm -> Sublayer -> Add
    
    优势:
    1. 梯度直通路径 (residual 不经过 norm)
    2. 训练更稳定
    3. 不需要 warmup
    """
    return x + sublayer(norm(x))


# 梯度流分析
def analyze_gradient_flow():
    """
    Pre-Norm 的梯度流分析
    """
    print("""
    Post-Norm 梯度流:
    ∂L/∂x = ∂L/∂y · ∂Norm/∂(x + sublayer(x))
    → 梯度必须经过 Norm，可能被缩放或消失
    
    Pre-Norm 梯度流:
    ∂L/∂x = ∂L/∂y · (∂x/∂x + ∂sublayer(Norm(x))/∂x)
          = ∂L/∂y · (1 + ...)
    → 有恒等路径，梯度可以直接回传
    
    这就是为什么 Pre-Norm 更稳定！
    """)
```

---

## 完整 Transformer 实现

### 📝 GPT-style LLM

```python
class GPTModel(nn.Module):
    """
    完整的 GPT-style 语言模型
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = None,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_rope: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Position Encoding (如果不用 RoPE)
        if not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.use_rope = use_rope
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            GPTDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = RMSNorm(d_model)
        
        # LM Head (共享 embedding 权重)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list = None
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: [batch, seq_len]
            kv_caches: list of KV cache for each layer
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_caches: updated caches
        """
        batch_size, seq_len = input_ids.shape
        
        # Token Embedding
        x = self.token_emb(input_ids)  # [batch, seq, d_model]
        
        # Position Embedding (如果不用 RoPE)
        if not self.use_rope:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_emb(positions)
        
        # Causal Mask
        # 只在 prefill 时需要，decode 时每次只有 1 个 token
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
        else:
            causal_mask = None
        
        # 初始化 KV Cache（如果需要）
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        
        new_kv_caches = []
        
        # Transformer Layers
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, mask=causal_mask, kv_cache=kv_caches[i])
            new_kv_caches.append(new_cache)
        
        # Final Norm
        x = self.final_norm(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        return logits, new_kv_caches
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        自回归生成
        """
        generated = input_ids.clone()
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # 只输入最后一个 token（使用 KV Cache）
            if kv_caches is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            # 前向
            logits, kv_caches = self.forward(curr_input, kv_caches)
            next_token_logits = logits[:, -1, :]
            
            # 采样
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


# ==================== 模型配置示例 ====================
MODEL_CONFIGS = {
    "gpt2-small": {
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    "llama-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,  # MHA
        "d_ff": 11008,
    },
    "llama2-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "d_ff": 11008,
    },
    "mistral-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,  # GQA
        "d_ff": 14336,
    },
}
```

---

## 面试追问汇总

### 架构相关

| 问题 | 答案 |
|:---|:---|
| Encoder-Decoder vs Decoder-Only | Decoder-Only 更简单，训练更高效 |
| 为什么 LLM 用 Decoder-Only | 自回归生成天然适配，不需要双向信息 |
| FFN 的作用 | 存储知识，增加模型容量 |

### FFN 相关

| 问题 | 答案 |
|:---|:---|
| FFN 维度为什么是 4x | 经验值，增加容量 |
| SwiGLU vs GELU | SwiGLU 效果更好，但参数多 50% |
| FFN 占总参数比例 | 约 2/3（远大于 Attention） |

### 层归一化

| 问题 | 答案 |
|:---|:---|
| Pre-Norm vs Post-Norm | Pre-Norm 梯度更稳定 |
| 为什么用 RMSNorm | 比 LayerNorm 快，效果相当 |
| 最终层需要 Norm 吗 | Pre-Norm 架构需要额外的 final_norm |

---

## 🔗 相关题目

- [Multi-Head Attention](01-attention.md) - Transformer 核心组件
- [RMSNorm](02-normalization.md#rmsnorm) - 现代 LLM 使用的归一化
- [RoPE](03-position-encoding.md#rope) - 现代位置编码
- [KV Cache](09-inference-optimization.md#kv-cache) - 推理加速
