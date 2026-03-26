# ⚡ 推理优化

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

推理优化决定 LLM 服务的成本和体验。KV Cache 是必考题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [KV Cache](#kv-cache)
- [PagedAttention](#pagedattention)
- [Speculative Decoding](#speculative-decoding)
- [Continuous Batching](#continuous-batching)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：推理优化的核心目标是**提高吞吐量**和**降低延迟**

| 方法 | 解决问题 | 核心思想 | 效果 |
|:---|:---|:---|:---|
| **KV Cache** | 重复计算 | 缓存历史 K/V | 计算减少 n 倍 |
| **PagedAttention** | 显存碎片 | 分页管理显存 | 显存利用率提升 |
| **Speculative Decoding** | 生成太慢 | 小模型推测+大模型验证 | 加速 2-3x |
| **Continuous Batching** | batch 利用率低 | 动态插入新请求 | 吞吐量提升 2-5x |

```python
# 推理的两个阶段，优化点不同！

# 1. Prefill（首次，compute-bound）
# - 处理整个 prompt，生成第一个 token
# - 矩阵乘法多，GPU 利用率高
# - 优化: 并行、量化

# 2. Decode（递增，memory-bound）
# - 每次只生成 1 个 token，但要读取整个 KV Cache
# - 维度小，内存带宽是瓶颈
# - 优化: KV Cache、batching、speculative
```

> 🤔 **Q: 为什么 Decode 阶段是 memory-bound？**
>
> Decode 时，每次只生成 1 个 token（维度 [1, hidden]），
> 但要读取整个 KV Cache（维度 [seq_len, hidden]）。
>
> 计算量很小，但内存读写很大，所以是内存带宽瓶颈。

---

## KV Cache

### 🎯 核心思想

自回归生成时，缓存历史 token 的 Key 和 Value，避免重复计算。

**没有 KV Cache**：生成第 n 个 token 需要计算 n 次 K, V
**有 KV Cache**：只计算新 token 的 K, V，复用历史缓存

> 🤔 **Q: KV Cache 占多少显存？怎么估算？**
>
> 公式：`2 × num_layers × batch × seq_len × num_kv_heads × head_dim × dtype_bytes`
>
> 例如 LLaMA-2 7B（FP16）：
> - 32 layers, 32 heads, head_dim=128, seq=4096, batch=1
> - = 2 × 32 × 1 × 4096 × 32 × 128 × 2 bytes = **2 GB**
>
> 用 GQA 的话（num_kv_heads=8），只需 **0.5 GB**！

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCache:
    """
    KV Cache 🔥🔥🔥 面试必考
    
    缓存 Attention 的 Key 和 Value，避免重复计算
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 预分配最大空间
        # 形状: [batch, num_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        # 当前缓存长度
        self.current_len = 0
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple:
        """
        更新缓存并返回完整的 K, V
        
        Args:
            k_new: [batch, num_heads, new_seq_len, head_dim] 新的 Key
            v_new: [batch, num_heads, new_seq_len, head_dim] 新的 Value
        
        Returns:
            k_full: [batch, num_heads, current_len + new_len, head_dim]
            v_full: 同上
        """
        new_len = k_new.size(2)
        
        # 写入新位置
        self.k_cache[:, :, self.current_len:self.current_len + new_len, :] = k_new
        self.v_cache[:, :, self.current_len:self.current_len + new_len, :] = v_new
        
        self.current_len += new_len
        
        # 返回有效部分
        k_full = self.k_cache[:, :, :self.current_len, :]
        v_full = self.v_cache[:, :, :self.current_len, :]
        
        return k_full, v_full
    
    def get_seq_len(self) -> int:
        """获取当前缓存长度"""
        return self.current_len
    
    def clear(self):
        """清空缓存"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0


class AttentionWithKVCache(nn.Module):
    """
    带 KV Cache 的 Attention
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache = None,
        use_cache: bool = True
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Args:
            x: [batch, seq_len, d_model]
               - Prefill 阶段: seq_len 是整个 prompt 长度
               - Decode 阶段: seq_len = 1（每次只处理新 token）
            kv_cache: KV 缓存对象
            use_cache: 是否使用缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用 KV Cache
        if use_cache and kv_cache is not None:
            K, V = kv_cache.update(K, V)
        
        # Attention 计算
        # Q: [batch, heads, seq_q, head_dim] （seq_q 可能只有 1）
        # K, V: [batch, heads, seq_k, head_dim] （seq_k 包含历史）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask（只在 prefill 时需要，decode 时 Q 只有一个 token）
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, K.size(2), device=x.device),
                diagonal=K.size(2) - seq_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, kv_cache


# ==================== 内存分析 ====================
def analyze_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16
):
    """分析 KV Cache 内存占用"""
    
    # 每层的 KV Cache: 2 * batch * heads * seq * head_dim
    per_layer = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    total = num_layers * per_layer
    
    print(f"=== KV Cache 内存分析 ===")
    print(f"配置: batch={batch_size}, seq={seq_len}, layers={num_layers}")
    print(f"      heads={num_heads}, head_dim={head_dim}")
    print(f"每层 KV Cache: {per_layer / 1024**2:.2f} MB")
    print(f"总 KV Cache: {total / 1024**3:.2f} GB")
    
    # 对比模型参数量
    model_params = num_layers * (4 * num_heads * head_dim * num_heads * head_dim)  # 简化估计
    print(f"对比：模型参数约 {model_params * dtype_bytes / 1024**3:.2f} GB")


if __name__ == "__main__":
    # LLaMA-7B 配置
    analyze_kv_cache_memory(
        batch_size=1,
        seq_len=4096,
        num_layers=32,
        num_heads=32,
        head_dim=128
    )
```

### 💡 面试追问

**Q: 为什么 KV Cache 能加速？**

> 自回归生成时，历史 token 的 K, V 不变。没有缓存每次要重算 O(n²)，有缓存只需 O(n)。

**Q: KV Cache 的内存瓶颈是什么？**

> 内存随 batch_size × seq_len × num_layers 线性增长。长序列或大 batch 时成为瓶颈。

---

## PagedAttention

### 🎯 核心思想

借鉴操作系统的虚拟内存：将 KV Cache 分成固定大小的"页"，按需分配。

**问题**：预分配连续内存浪费（max_seq_len 通常用不满）
**解决**：分页管理，只分配实际需要的页

### 📝 实现代码

```python
class PagedKVCache:
    """
    Paged KV Cache (vLLM 核心技术)
    
    将 KV Cache 分成固定大小的页，按需分配
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,  # 每页存储的 token 数
        max_pages: int = 1000,  # 最大页数
        dtype: torch.dtype = torch.float16,
        device: torch.device = None
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.device = device
        
        # 页表：page_id -> (k_page, v_page)
        # 实际实现中会预分配所有页，这里简化
        self.pages = {}
        
        # 空闲页列表
        self.free_pages = list(range(max_pages))
        
        # 每个序列的页表：seq_id -> [page_ids]
        self.seq_page_table = {}
    
    def allocate_page(self) -> int:
        """分配一个新页"""
        if not self.free_pages:
            raise RuntimeError("Out of pages!")
        
        page_id = self.free_pages.pop(0)
        
        # 分配实际内存
        self.pages[page_id] = {
            'k': torch.zeros(
                self.num_layers, self.num_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device
            ),
            'v': torch.zeros(
                self.num_layers, self.num_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device
            ),
            'used_slots': 0
        }
        
        return page_id
    
    def free_page(self, page_id: int):
        """释放页"""
        if page_id in self.pages:
            del self.pages[page_id]
            self.free_pages.append(page_id)
    
    def append_token(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        为序列追加一个 token 的 KV
        
        k, v: [num_heads, 1, head_dim]
        """
        if seq_id not in self.seq_page_table:
            self.seq_page_table[seq_id] = []
        
        page_ids = self.seq_page_table[seq_id]
        
        # 检查是否需要新页
        if not page_ids or self.pages[page_ids[-1]]['used_slots'] >= self.page_size:
            new_page_id = self.allocate_page()
            page_ids.append(new_page_id)
        
        # 写入最后一页
        current_page_id = page_ids[-1]
        slot_idx = self.pages[current_page_id]['used_slots']
        
        self.pages[current_page_id]['k'][layer_idx, :, slot_idx, :] = k.squeeze(1)
        self.pages[current_page_id]['v'][layer_idx, :, slot_idx, :] = v.squeeze(1)
        self.pages[current_page_id]['used_slots'] += 1
    
    def get_kv(self, seq_id: int, layer_idx: int) -> tuple:
        """
        获取序列的完整 KV
        
        Returns:
            k: [num_heads, seq_len, head_dim]
            v: [num_heads, seq_len, head_dim]
        """
        if seq_id not in self.seq_page_table:
            return None, None
        
        page_ids = self.seq_page_table[seq_id]
        k_list, v_list = [], []
        
        for page_id in page_ids:
            page = self.pages[page_id]
            used = page['used_slots']
            
            k_list.append(page['k'][layer_idx, :, :used, :])
            v_list.append(page['v'][layer_idx, :, :used, :])
        
        k = torch.cat(k_list, dim=1)  # [heads, total_len, head_dim]
        v = torch.cat(v_list, dim=1)
        
        return k, v
    
    def free_sequence(self, seq_id: int):
        """释放序列的所有页"""
        if seq_id in self.seq_page_table:
            for page_id in self.seq_page_table[seq_id]:
                self.free_page(page_id)
            del self.seq_page_table[seq_id]


def paged_attention_advantages():
    """PagedAttention 的优势"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              PagedAttention 优势                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  1. 内存利用率提升                                           ║
    ║     - 传统: 预分配 max_seq_len，浪费 50-90%                  ║
    ║     - Paged: 按需分配，接近 0 浪费                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  2. 支持更大的 batch size                                    ║
    ║     - 相同内存下，batch size 可增加 2-4x                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  3. 灵活的内存共享                                           ║
    ║     - Beam Search: 多个候选共享相同 prefix 的页              ║
    ║     - Parallel Sampling: 共享 prompt 的 KV                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  4. 高效的内存回收                                           ║
    ║     - 序列结束后立即释放页                                   ║
    ║     - 不需要等待整个 batch 完成                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
```

---

## Speculative Decoding

### 🎯 核心思想

用小模型"猜"多个 token，大模型验证。如果猜对了，相当于一次前向生成多个 token。

**加速比**：理论上可达 2-3x（取决于小模型的准确率）

### 📝 实现代码

```python
def speculative_decode(
    draft_model,      # 小模型（快但不那么准）
    target_model,     # 大模型（慢但准）
    input_ids: torch.Tensor,
    gamma: int = 4,   # 每次猜几个 token
    max_new_tokens: int = 100
):
    """
    Speculative Decoding 🔥🔥
    
    原理：
    1. 用小模型快速生成 γ 个 token（草稿）
    2. 用大模型一次性验证这 γ 个 token
    3. 接受部分或全部草稿，拒绝的位置从大模型重采样
    
    为什么快？
    - 大模型的瓶颈是内存带宽而非计算
    - 验证 γ 个 token 和生成 1 个 token 耗时接近
    - 如果草稿全部正确，一次前向 = γ 个 token
    """
    device = input_ids.device
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens // gamma):
        # ========== Step 1: 小模型生成草稿 ==========
        draft_tokens = []
        draft_probs = []
        
        draft_input = generated.clone()
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(token)
                draft_probs.append(probs)
                draft_input = torch.cat([draft_input, token], dim=-1)
        
        draft_tokens = torch.cat(draft_tokens, dim=-1)  # [batch, gamma]
        
        # ========== Step 2: 大模型验证 ==========
        # 一次性计算所有位置的分布
        verify_input = torch.cat([generated, draft_tokens], dim=-1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # 取对应位置的分布
        # target_logits[:, generated_len-1] 预测第一个草稿位置
        # target_logits[:, generated_len] 预测第二个草稿位置
        # ...
        start_pos = generated.size(1) - 1
        target_probs_list = [
            F.softmax(target_logits[:, start_pos + i, :], dim=-1)
            for i in range(gamma + 1)
        ]
        
        # ========== Step 3: 接受/拒绝 ==========
        accepted_count = 0
        
        for i in range(gamma):
            draft_token = draft_tokens[:, i]
            
            # 获取草稿和目标分布
            p_draft = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            p_target = target_probs_list[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            
            # 接受概率: min(1, p_target / p_draft)
            accept_prob = torch.clamp(p_target / p_draft, max=1.0)
            
            if torch.rand(1, device=device).item() < accept_prob.item():
                # 接受，添加这个 token
                generated = torch.cat([generated, draft_token.unsqueeze(1)], dim=-1)
                accepted_count += 1
            else:
                # 拒绝，从调整后的分布采样
                # 调整分布: max(0, p_target - p_draft) 然后归一化
                adjusted_probs = torch.clamp(
                    target_probs_list[i] - draft_probs[i],
                    min=0
                )
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
                
                new_token = torch.multinomial(adjusted_probs, num_samples=1)
                generated = torch.cat([generated, new_token], dim=-1)
                break
        
        # 如果全部接受，从大模型最后的分布额外采样一个
        if accepted_count == gamma:
            bonus_token = torch.multinomial(target_probs_list[-1], num_samples=1)
            generated = torch.cat([generated, bonus_token], dim=-1)
    
    return generated


def analyze_speculative_speedup(draft_accuracy: float, gamma: int):
    """分析 Speculative Decoding 的加速比"""
    
    # 期望每次验证接受的 token 数
    # 如果每个草稿独立，期望 = sum(accuracy^i for i in 1..gamma)
    expected_accepted = sum(draft_accuracy ** i for i in range(1, gamma + 1))
    
    # 考虑全部正确时的 bonus token
    expected_with_bonus = expected_accepted + draft_accuracy ** gamma
    
    # 加速比 = 期望生成的 token 数 / 大模型前向次数
    speedup = expected_with_bonus / 1  # 每次大模型只前向一次
    
    print(f"=== Speculative Decoding 加速分析 ===")
    print(f"草稿模型准确率: {draft_accuracy:.0%}")
    print(f"每次猜测数量 γ: {gamma}")
    print(f"期望接受的 token: {expected_accepted:.2f}")
    print(f"期望生成的 token (含 bonus): {expected_with_bonus:.2f}")
    print(f"理论加速比: {speedup:.2f}x")


if __name__ == "__main__":
    analyze_speculative_speedup(draft_accuracy=0.8, gamma=4)
```

---

## Continuous Batching

### 🎯 核心思想

序列完成后立即加入新请求，而不是等整个 batch 完成。

**问题**：静态 batching 中，短序列要等长序列完成
**解决**：动态管理 batch，序列独立调度

### 📝 实现代码

```python
class ContinuousBatchingScheduler:
    """
    Continuous Batching 调度器
    
    核心思想：
    - 每个序列独立跟踪状态
    - 序列完成后立即移出，新序列立即加入
    - 最大化 GPU 利用率
    """
    
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        
        # 活跃序列
        self.active_sequences = {}  # seq_id -> SequenceState
        
        # 等待队列
        self.waiting_queue = []
    
    def add_request(self, request):
        """添加新请求到等待队列"""
        self.waiting_queue.append(request)
    
    def schedule_step(self):
        """
        调度一个推理步骤
        
        Returns:
            batch: 本步骤要处理的序列
        """
        # 移除已完成的序列
        completed = [
            seq_id for seq_id, state in self.active_sequences.items()
            if state.is_finished()
        ]
        for seq_id in completed:
            del self.active_sequences[seq_id]
        
        # 从等待队列添加新序列（填满 batch）
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            request = self.waiting_queue.pop(0)
            seq_id = self.create_sequence(request)
            self.active_sequences[seq_id] = SequenceState(request)
        
        # 返回当前 batch
        return list(self.active_sequences.values())
    
    def create_sequence(self, request):
        """创建新序列"""
        return id(request)  # 简化：用对象 id 作为 seq_id


class SequenceState:
    """序列状态"""
    
    def __init__(self, request):
        self.request = request
        self.generated_tokens = []
        self.max_tokens = request.get('max_tokens', 100)
    
    def is_finished(self):
        """检查是否完成"""
        if len(self.generated_tokens) >= self.max_tokens:
            return True
        if self.generated_tokens and self.generated_tokens[-1] == EOS_TOKEN:
            return True
        return False
    
    def add_token(self, token):
        """添加生成的 token"""
        self.generated_tokens.append(token)


EOS_TOKEN = 2  # 假设的结束符
```

---

## 面试追问汇总

### KV Cache 相关

| 问题 | 答案 |
|:---|:---|
| 内存计算公式 | 2 × batch × layers × heads × seq × head_dim × dtype_bytes |
| 为什么能加速 | 避免重复计算历史 token 的 K, V |
| 瓶颈是什么 | 内存带宽而非计算 |

### PagedAttention 相关

| 问题 | 答案 |
|:---|:---|
| 核心思想 | 借鉴虚拟内存，分页管理 KV Cache |
| 优势 | 内存利用率高，支持更大 batch |
| vLLM 用什么技术 | PagedAttention + Continuous Batching |

### Speculative Decoding 相关

| 问题 | 答案 |
|:---|:---|
| 为什么能加速 | 大模型验证和生成单 token 耗时接近 |
| 加速比取决于什么 | 小模型准确率、γ 大小 |
| 典型加速比 | 2-3x |

---

## 🔗 相关题目

- [Multi-Head Attention](01-attention.md) - KV Cache 的基础
- [GQA/MQA](01-attention.md#gqa) - 减少 KV Cache 内存
- [Flash Attention](01-attention.md#flash-attention) - 加速 Attention 计算
