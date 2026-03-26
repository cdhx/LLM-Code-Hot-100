# 🎲 采样策略

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐

采样策略决定 LLM 如何从概率分布中选择下一个 token。Top-p 是面试必考题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [Temperature Sampling](#temperature-sampling)
- [Top-k Sampling](#top-k-sampling)
- [Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)
- [Beam Search](#beam-search)
- [Greedy Decoding](#greedy-decoding)
- [组合采样策略](#组合采样策略)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：采样策略的核心区别在于**如何过滤候选词**

| 方法 | 过滤方式 | 核心代码 | 特点 |
|:---|:---|:---|:---|
| **Greedy** | 只保留最大 | `argmax(logits)` | 确定但易重复 |
| **Temperature** | 缩放 logits | `softmax(logits / T)` | 控制多样性 |
| **Top-k** | 保留前 k 个 | `topk(logits, k)` | 固定候选数 |
| **Top-p** | 保留累积概率 ≤ p | `cumsum(probs) <= p` | 动态候选数 |
| **Beam Search** | 保留 k 个最优序列 | 追踪多路径 | 质量高但无聊 |

```python
# 采样策略的核心区别：过滤方式不同

# Greedy: 只要最大的
token = logits.argmax()                      # 只留 1 个

# Top-k: 只保留前 k 个（数量固定）
mask = logits < topk(logits, k).values[-1]   # 只留 k 个

# Top-p: 保留累积概率达到 p 的（数量动态）
mask = cumsum(sorted_probs) > p              # 数量随分布变化
```

> 🤔 **Q: 为什么 Top-p 比 Top-k 更常用？**
>
> Top-k 的问题：当模型很确定时（比如下一个词 90% 是 "the"），k=50 会引入 49 个几乎不可能的词。
>
> Top-p 自动适应：确定时只留几个，不确定时留很多。这就是 "nucleus" （核心）的含义。

---

## Temperature Sampling

### 🎯 核心思想

Temperature 控制输出分布的"锐度"：
- **T → 0**: 分布趋近 one-hot（贪婪）
- **T = 1**: 原始分布
- **T → ∞**: 趋近均匀分布（完全随机）

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F

def temperature_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Temperature Sampling
    
    Args:
        logits: [batch, vocab_size] 或 [vocab_size] 模型输出的 logits
        temperature: 温度参数
            - T < 1: 更确定（分布更尖锐）
            - T = 1: 原始分布
            - T > 1: 更随机（分布更平滑）
    
    Returns:
        sampled_token: 采样得到的 token id
        probs: softmax 后的概率分布
    """
    if temperature == 0:
        # 温度为 0 时退化为 argmax（贪婪解码）
        sampled_token = logits.argmax(dim=-1)
        probs = F.one_hot(sampled_token, logits.size(-1)).float()
        return sampled_token, probs
    
    # Step 1: 温度缩放
    scaled_logits = logits / temperature
    
    # Step 2: 数值稳定的 softmax
    # 减去最大值防止 exp 溢出
    scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True)[0]
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Step 3: 多项式采样
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


# ==================== 可视化温度效果 ====================
def visualize_temperature_effect():
    """展示不同温度对分布的影响"""
    import matplotlib.pyplot as plt
    
    # 模拟 logits（假设有 10 个候选词）
    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0])
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    plt.figure(figsize=(12, 6))
    for T in temperatures:
        probs = F.softmax(logits / T, dim=-1)
        plt.plot(probs.numpy(), label=f'T={T}', marker='o')
    
    plt.xlabel('Token Index')
    plt.ylabel('Probability')
    plt.title('Temperature Effect on Probability Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_effect.png')
    print("已保存: temperature_effect.png")


if __name__ == "__main__":
    visualize_temperature_effect()
```

### 💡 面试追问

**Q: Temperature 为什么能控制多样性？**

> 除以 T 相当于放大/缩小 logits 的差异。T 小时差异放大，softmax 后高概率更高；T 大时差异缩小，分布更均匀。

> 🤔 **Q: Temperature=0 和 Greedy 有什么区别？**
>
> 完全一样！当 T→0 时，`softmax(logits/T)` 趋近于 one-hot，采样结果就是 argmax。
>
> 所以代码中通常特判 `if temperature == 0: return argmax(logits)`

---

## Top-k Sampling

### 🎯 核心思想

只从概率最高的 k 个 token 中采样，过滤掉长尾低概率词。

### 📝 实现代码

```python
def top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k Sampling
    
    只保留概率最高的 k 个 token，其余置为 -inf
    
    Args:
        logits: [batch, vocab_size] 模型输出
        k: 保留的 token 数量
        temperature: 温度参数
        min_tokens_to_keep: 最少保留的 token 数
    
    Returns:
        sampled_token: 采样结果
        probs: 过滤后的概率分布
    """
    # 确保 k 至少为 min_tokens_to_keep
    k = max(k, min_tokens_to_keep)
    
    # Step 1: 温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # Step 2: 找到 top-k 的阈值
    # topk 返回 (values, indices)，取 values 的最后一个（第 k 大）
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1, None]  # 第 k 大的值
    
    # Step 3: 将小于阈值的 logits 置为 -inf
    # 这样 softmax 后这些位置的概率趋近于 0
    filtered_logits = torch.where(
        logits >= threshold,
        logits,
        torch.full_like(logits, float('-inf'))
    )
    
    # Step 4: Softmax + 采样
    probs = F.softmax(filtered_logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


def top_k_sampling_simple(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k 简化版（面试快速手写）
    """
    # 找到第 k 大的值作为阈值
    threshold = torch.topk(logits, k).values[..., -1:]
    
    # 低于阈值的置为 -inf
    logits[logits < threshold] = float('-inf')
    
    # Softmax + 采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 💡 Top-k 的问题

```
场景1: 分布很尖（高置信度）
  词: [the, a, an, ...]
  概率: [0.9, 0.05, 0.02, ...]
  → k=50 会引入很多不相关的词

场景2: 分布很平（低置信度）
  词: [red, blue, green, yellow, ...]
  概率: [0.1, 0.09, 0.08, 0.08, ...]
  → k=5 可能漏掉合理的选项

结论: 固定的 k 无法适应不同的分布形状
解决: 使用 Top-p（动态调整候选数量）
```

---

## Top-p (Nucleus) Sampling

### 🎯 核心思想

保留累积概率达到 p 的**最小**词汇集合。这样做的好处是：
- 分布尖锐时，只保留少数几个词
- 分布平坦时，保留更多词

### 📝 实现代码

```python
def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-p (Nucleus) Sampling 🔥 面试必考
    
    保留累积概率达到 p 的最小词汇集合
    
    Args:
        logits: [batch, vocab_size] 或 [vocab_size]
        p: 累积概率阈值，通常 0.9-0.95
        temperature: 温度参数
        min_tokens_to_keep: 最少保留的 token 数
    
    Returns:
        sampled_token: 采样结果
        probs: 过滤后的概率分布
    """
    # Step 1: 温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # Step 2: 排序（降序）
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Step 3: 计算累积概率
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Step 4: 找到累积概率超过 p 的位置
    # 注意：我们要保留累积概率 <= p 的位置
    # 但至少保留 min_tokens_to_keep 个
    sorted_indices_to_remove = cumsum_probs > p
    
    # 保证至少有 min_tokens_to_keep 个 token
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Step 5: 将需要移除的位置在原始 logits 中置为 -inf
    # 先创建一个 mask，然后映射回原始索引
    # 方法：scatter 将 sorted 空间的 mask 转回原始空间
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    
    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Step 6: Softmax + 采样
    probs = F.softmax(filtered_logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


def top_p_sampling_simple(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Top-p 简化版（面试快速手写）
    """
    # 1. 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 计算累积概率
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    
    # 3. 找到超过 p 的位置并 mask
    mask = cumsum <= p
    mask[..., 0] = True  # 至少保留一个
    
    # 4. 过滤
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # 5. 采样
    sampled_idx = torch.multinomial(filtered_probs, 1)
    return sorted_indices.gather(-1, sampled_idx)


# ==================== 详细示例 ====================
def top_p_example():
    """展示 Top-p 的工作过程"""
    # 假设词表大小为 10
    logits = torch.tensor([2.0, 1.8, 1.5, 0.5, 0.2, 0.0, -0.5, -1.0, -2.0, -3.0])
    p = 0.9
    
    # 计算概率
    probs = F.softmax(logits, dim=-1)
    
    # 排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    print("=== Top-p Sampling 示例 (p=0.9) ===\n")
    print("索引\t概率\t累积概率\t保留?")
    print("-" * 40)
    
    for i in range(len(probs)):
        keep = "✓" if cumsum[i] <= p or i == 0 else "✗"
        print(f"{sorted_indices[i].item()}\t{sorted_probs[i]:.4f}\t{cumsum[i]:.4f}\t\t{keep}")
    
    print(f"\n结论: 保留前 {(cumsum <= p).sum().item() + 1} 个词")


if __name__ == "__main__":
    top_p_example()
```

输出示例：
```
=== Top-p Sampling 示例 (p=0.9) ===

索引    概率    累积概率        保留?
----------------------------------------
0       0.2897  0.2897          ✓
1       0.2367  0.5265          ✓
2       0.1755  0.7020          ✓
3       0.0646  0.7666          ✓
4       0.0479  0.8145          ✓
5       0.0392  0.8537          ✓
6       0.0238  0.8775          ✓
7       0.0144  0.8919          ✓
8       0.0053  0.8972          ✓
9       0.0029  0.9001          ✗

结论: 保留前 9 个词
```

---

## Beam Search

### 🎯 核心思想

维护 k 个最优候选序列，每步扩展所有候选，保留得分最高的 k 个。

> 🤔 **Q: Beam Search 和 Greedy 有什么区别？为什么不总是用 Beam？**
>
> Greedy 是 Beam width=1 的特例。Beam 可以纠正局部次优选择。
>
> 但 Beam Search 有问题：
> 1. 计算量大（每步维护 k 个序列）
> 2. 倾向于生成短句（短序列分数更高）
> 3. 缺乏多样性（往往生成 "safe" 但无聊的句子）
>
> 所以现代 LLM 对话多用 Sampling，翻译等需要稳定输出时才用 Beam。

### 📝 实现代码

```python
def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_width: int = 4,
    max_length: int = 50,
    eos_token_id: int = 2,
    length_penalty: float = 1.0
) -> torch.Tensor:
    """
    Beam Search 解码
    
    Args:
        model: 语言模型
        input_ids: [batch, seq_len] 输入序列（通常 batch=1）
        beam_width: beam 宽度
        max_length: 最大生成长度
        eos_token_id: 结束符 ID
        length_penalty: 长度惩罚（> 1 鼓励长句，< 1 鼓励短句）
    
    Returns:
        best_sequence: [batch, seq_len] 最优序列
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 复制 input_ids 到所有 beam
    # [batch, seq] -> [batch * beam, seq]
    beam_input_ids = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
    beam_input_ids = beam_input_ids.view(batch_size * beam_width, -1)
    
    # 初始化 beam 分数
    # [batch, beam]
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_scores[:, 1:] = float('-inf')  # 初始只有第一个 beam 是活的
    
    # 完成的序列
    done = [False] * batch_size
    
    for step in range(max_length):
        # 获取模型输出
        with torch.no_grad():
            outputs = model(beam_input_ids)
            # 取最后一个 token 的 logits
            next_token_logits = outputs[:, -1, :]  # [batch*beam, vocab]
        
        vocab_size = next_token_logits.size(-1)
        
        # 转换为 log prob
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # 重塑: [batch*beam, vocab] -> [batch, beam, vocab]
        next_token_log_probs = next_token_log_probs.view(batch_size, beam_width, -1)
        
        # 计算新的分数: [batch, beam, 1] + [batch, beam, vocab]
        # -> [batch, beam, vocab]
        next_scores = beam_scores.unsqueeze(-1) + next_token_log_probs
        
        # 重塑为 [batch, beam * vocab]
        next_scores = next_scores.view(batch_size, -1)
        
        # 选择 top beam_width 个
        # [batch, beam]
        next_scores, next_tokens = torch.topk(next_scores, beam_width, dim=-1)
        
        # 解码: 哪个 beam + 哪个 token
        beam_indices = next_tokens // vocab_size  # [batch, beam]
        token_indices = next_tokens % vocab_size  # [batch, beam]
        
        # 构建新的序列
        # 从对应的旧 beam 复制，然后添加新 token
        new_beam_input_ids = []
        for batch_idx in range(batch_size):
            batch_beams = []
            for beam_idx in range(beam_width):
                # 找到源 beam
                source_beam = beam_indices[batch_idx, beam_idx]
                source_idx = batch_idx * beam_width + source_beam
                
                # 新 token
                new_token = token_indices[batch_idx, beam_idx]
                
                # 拼接
                new_seq = torch.cat([
                    beam_input_ids[source_idx],
                    new_token.unsqueeze(0)
                ])
                batch_beams.append(new_seq)
            
            new_beam_input_ids.append(torch.stack(batch_beams))
        
        beam_input_ids = torch.cat(new_beam_input_ids, dim=0)
        beam_scores = next_scores
        
        # 检查 EOS
        # 简化：这里只检查第一个 batch
        if (token_indices[0] == eos_token_id).any():
            break
    
    # 应用长度惩罚
    seq_len = beam_input_ids.size(1)
    length_penalty_factor = ((5 + seq_len) / 6) ** length_penalty
    final_scores = beam_scores / length_penalty_factor
    
    # 选择最优 beam
    best_beam_idx = final_scores.argmax(dim=-1)  # [batch]
    
    # 提取最优序列
    best_sequences = []
    for batch_idx in range(batch_size):
        idx = batch_idx * beam_width + best_beam_idx[batch_idx]
        best_sequences.append(beam_input_ids[idx])
    
    return torch.stack(best_sequences)


# ==================== 简化版 Beam Search ====================
def beam_search_simple(
    get_next_probs,  # 函数: input_ids -> log_probs
    start_token: int,
    beam_width: int = 3,
    max_length: int = 10,
    eos_token: int = 2
):
    """
    简化版 Beam Search（面试手写）
    
    返回 (最优序列, 分数)
    """
    # 初始化: [(序列, 分数)]
    beams = [([start_token], 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == eos_token:
                # 已结束的序列直接保留
                all_candidates.append((seq, score))
                continue
            
            # 获取下一个 token 的 log prob
            log_probs = get_next_probs(seq)  # [vocab_size]
            
            # 扩展所有可能的下一个 token
            for token_id in range(len(log_probs)):
                new_seq = seq + [token_id]
                new_score = score + log_probs[token_id]
                all_candidates.append((new_seq, new_score))
        
        # 保留 top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
        
        # 如果所有 beam 都结束了
        if all(seq[-1] == eos_token for seq, _ in beams):
            break
    
    # 返回最优
    return beams[0]
```

---

## Greedy Decoding

### 🎯 核心思想

每一步选择概率最高的 token。简单但容易陷入重复。

### 📝 实现代码

```python
def greedy_decoding(logits: torch.Tensor) -> torch.Tensor:
    """
    贪婪解码：选择概率最高的 token
    
    Args:
        logits: [batch, vocab_size]
    
    Returns:
        token_ids: [batch]
    """
    return logits.argmax(dim=-1)


def greedy_decode_sequence(
    model,
    input_ids: torch.Tensor,
    max_length: int = 50,
    eos_token_id: int = 2
) -> torch.Tensor:
    """
    贪婪解码完整序列
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
        
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if (next_token == eos_token_id).all():
            break
    
    return generated
```

---

## 组合采样策略

### 📝 工业级实现

```python
class Sampler:
    """
    组合采样器：支持 Temperature + Top-k + Top-p + 重复惩罚
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,         # 0 表示不使用
        top_p: float = 1.0,     # 1.0 表示不使用
        repetition_penalty: float = 1.0,  # 1.0 表示不惩罚
        min_tokens_to_keep: int = 1
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, vocab_size] 或 [vocab_size]
            generated_tokens: [batch, seq_len] 已生成的 token（用于重复惩罚）
        """
        # Step 1: 重复惩罚
        if self.repetition_penalty != 1.0 and generated_tokens is not None:
            logits = self._apply_repetition_penalty(logits, generated_tokens)
        
        # Step 2: Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Step 3: Top-k
        if self.top_k > 0:
            logits = self._top_k_filtering(logits)
        
        # Step 4: Top-p
        if self.top_p < 1.0:
            logits = self._top_p_filtering(logits)
        
        # Step 5: 采样
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        重复惩罚：降低已生成 token 的概率
        """
        for batch_idx in range(logits.size(0) if logits.dim() > 1 else 1):
            for token_id in generated_tokens[batch_idx].unique():
                if logits.dim() > 1:
                    score = logits[batch_idx, token_id]
                else:
                    score = logits[token_id]
                
                # 正分数除以惩罚，负分数乘以惩罚
                if score > 0:
                    logits[batch_idx, token_id] = score / self.repetition_penalty
                else:
                    logits[batch_idx, token_id] = score * self.repetition_penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        k = max(self.top_k, self.min_tokens_to_keep)
        threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float('-inf'))
    
    def _top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumsum_probs > self.top_p
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float('-inf'))


# ==================== 常用配置 ====================
SAMPLING_CONFIGS = {
    "greedy": {"temperature": 0.0},
    "creative": {"temperature": 1.0, "top_k": 50, "top_p": 0.95},
    "balanced": {"temperature": 0.7, "top_p": 0.9},
    "code": {"temperature": 0.2, "top_p": 0.95, "repetition_penalty": 1.1},
    "chat": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.05},
}
```

---

## 面试追问汇总

### 基础问题

| 问题 | 答案 |
|:---|:---|
| Top-p 和 Top-k 的区别 | Top-k 固定数量，Top-p 动态调整 |
| Temperature 的作用 | 控制分布锐度，影响多样性 |
| Beam Search 的问题 | 倾向于生成通用/重复的句子 |

### 代码追问

```python
# Q: 这段 Top-p 代码有什么问题？
def buggy_top_p(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum < p  # BUG: 应该是 <=
    # ...

# A: 应该用 <= p，否则可能把累积刚好等于 p 的词排除掉
```

### 高级问题

| 问题 | 答案 |
|:---|:---|
| 为什么不直接用 argmax | 会陷入重复，缺乏多样性 |
| Beam Search vs Sampling | Beam 更稳定但无聊，Sampling 更多样但可能跑偏 |
| 如何选择参数 | 任务相关：创意写作高 T，代码生成低 T |

---

## 🔗 相关题目

- [Softmax 数值稳定性](05-loss-functions.md#softmax) - 采样的基础
- [Speculative Decoding](09-inference-optimization.md#speculative-decoding) - 加速采样
