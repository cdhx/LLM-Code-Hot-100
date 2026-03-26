# 🚀 高效训练

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

高效训练技术让我们能够训练更大的模型。LoRA 是面试必考题！

---

## 目录

- [方法一览对比](#方法一览对比)
- [LoRA](#lora)
- [Gradient Checkpointing](#gradient-checkpointing)
- [Mixed Precision Training](#mixed-precision-training)
- [Gradient Accumulation](#gradient-accumulation)
- [面试追问汇总](#面试追问汇总)

---

## 方法一览对比

> 💡 **一句话区分**：高效训练技术解决的是**显存不够**或**计算太慢**的问题

| 方法 | 解决问题 | 核心思想 | 代价 |
|:---|:---|:---|:---|
| **LoRA** | 参数太多 | 只训练低秩增量 | 效果略差 |
| **Gradient Checkpointing** | 激活值占显存 | 不存中间层，重算 | 计算多 30% |
| **Mixed Precision** | 模型+梯度占显存 | FP16 计算，FP32 主副本 | 需 loss scaling |
| **Gradient Accumulation** | batch 太小 | 累积多次梯度再更新 | 训练变慢 |

```python
# 显存占用分析（了解这个才能选对方法！）
# 假设 7B 模型，FP32，batch_size=1，seq_len=2048

# 1. 模型参数: 7B * 4 bytes = 28 GB
# 2. 梯度: 7B * 4 bytes = 28 GB
# 3. 优化器状态 (Adam): 7B * 2 * 4 bytes = 56 GB (m + v)
# 4. 激活值: 取决于 batch/seq，通常 10-50 GB

# 解决方案：
# - 参数太多 → LoRA（只训 0.1% 参数）
# - 激活值太大 → Gradient Checkpointing（不存，重算）
# - 模型+梯度大 → Mixed Precision（FP16，减半）
# - batch 太小 → Gradient Accumulation（累积多次）
```

> 🤔 **Q: 我显存不够，应该用哪个技术？**
>
> 1. 先试 Mixed Precision：最简单，一行代码减半显存
> 2. 再试 Gradient Checkpointing：再省 30-50%，但变慢
> 3. batch 小时用 Accumulation：等效大 batch
> 4. 参数量大时用 LoRA：只训练 0.1% 参数

---

## LoRA

### 🎯 核心思想

LoRA (Low-Rank Adaptation)：冻结预训练权重，只训练低秩分解的增量矩阵。

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

**参数量**：原始 $d \times k$ → LoRA $(d + k) \times r$

> 🤔 **Q: 为什么 B 初始化为 0，A 用 Kaiming？**
>
> 关键洞察：初始时 BA = 0，模型输出 = W + BA = W！
>
> 这样微调从**原模型出发**，而不是随机点。
>
> A 用 Kaiming 是保证梯度能流动（如果 A 也是 0，梯度为 0）。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA Layer 🔥 面试必考
    
    核心思想：
    - 冻结原始权重 W
    - 学习低秩增量 ΔW = BA
    - 输出 = Wx + BAx = Wx + B(Ax)
    
    参数量从 d*k 降低到 (d+k)*r
    当 r=8, d=k=4096 时，参数量降低 99.6%
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # 缩放因子：控制 LoRA 更新的幅度
        # alpha/rank 类似于学习率的作用
        self.scaling = lora_alpha / rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight.requires_grad = False  # 冻结
        
        # LoRA 可训练参数
        # A: [rank, in_features] - 降维
        # B: [out_features, rank] - 升维
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        初始化策略：
        - A: Kaiming 均匀初始化
        - B: 零初始化
        
        这样初始时 BA = 0，从原模型开始微调
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, in_features]
        """
        # 原始输出（冻结权重）
        original_output = F.linear(x, self.weight)
        
        # LoRA 输出
        # x @ A^T @ B^T * scaling
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # [batch, seq, rank]
        lora_output = F.linear(lora_output, self.lora_B)  # [batch, seq, out_features]
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        合并 LoRA 权重到原始权重（推理时使用）
        
        W' = W + scaling * BA
        """
        with torch.no_grad():
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
    
    def unmerge_weights(self):
        """取消合并（继续训练时使用）"""
        with torch.no_grad():
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)


class LoRALinear(nn.Module):
    """
    完整的 LoRA Linear 层（包含 bias）
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        self.original = original_linear
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # 冻结原始参数
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRA 参数
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_out = self.original(x)
        
        # LoRA 输出
        lora_out = self.dropout(x)
        lora_out = F.linear(F.linear(lora_out, self.lora_A), self.lora_B)
        
        return original_out + lora_out * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    target_modules: list = None
):
    """
    将 LoRA 应用到模型的指定层
    
    target_modules: 要替换的模块名称（如 ["q_proj", "v_proj"]）
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # 默认只对 Q, V 投影应用
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 创建 LoRA 层
                lora_layer = LoRALinear(module, rank=rank, lora_alpha=lora_alpha)
                
                # 替换原始层
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, lora_layer)
    
    return model


# ==================== 参数量分析 ====================
def analyze_lora_params(d: int, k: int, rank: int):
    """分析 LoRA 参数量"""
    original_params = d * k
    lora_params = (d + k) * rank
    ratio = lora_params / original_params * 100
    
    print(f"=== LoRA 参数分析 ===")
    print(f"原始参数: {d} x {k} = {original_params:,}")
    print(f"LoRA 参数 (rank={rank}): ({d} + {k}) x {rank} = {lora_params:,}")
    print(f"参数比例: {ratio:.2f}%")
    print(f"参数减少: {100 - ratio:.2f}%")


if __name__ == "__main__":
    # LLaMA-7B 的典型配置
    analyze_lora_params(d=4096, k=4096, rank=8)
```

### 💡 面试追问

**Q: 为什么 B 初始化为 0？**

> 保证训练开始时 ΔW = BA = 0，即从原模型开始。如果不是零初始化，初始就会偏离预训练模型。

**Q: 为什么只对 Q, V 应用 LoRA，不对 K 应用？**

> 实验发现对 Q, V 效果最好，对 K 效果较差。原因可能是 K 的变化会影响所有 Q 的 attention pattern。

---

## Gradient Checkpointing

### 🎯 核心思想

时间换空间：不保存中间激活值，反向传播时重新计算。

- **标准训练**: 保存所有层的激活 → 内存 O(n)
- **Checkpointing**: 只保存检查点，其他重算 → 内存 O(√n)，时间 +30%

### 📝 实现代码

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    """
    使用 Gradient Checkpointing 的 Transformer 层
    
    不保存中间激活值，反向传播时重新计算
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = True):
        if use_checkpoint and self.training:
            # 使用 checkpoint：不保存中间激活
            attn_out = checkpoint(
                self._attn_block,
                x,
                use_reentrant=False  # 推荐设为 False
            )
            ffn_out = checkpoint(
                self._ffn_block,
                x + attn_out,
                use_reentrant=False
            )
            return x + attn_out + ffn_out
        else:
            # 正常前向
            attn_out = self._attn_block(x)
            x = x + attn_out
            ffn_out = self._ffn_block(x)
            return x + ffn_out
    
    def _attn_block(self, x):
        """Attention 子块"""
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        return attn_out
    
    def _ffn_block(self, x):
        """FFN 子块"""
        return self.ffn(self.norm2(x))


def manual_checkpoint_forward(func, *args):
    """
    手动实现 Gradient Checkpointing
    
    原理：
    1. 前向时不保存中间结果（使用 no_grad）
    2. 反向时重新计算前向
    """
    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # 保存输入（用于反向时重算）
            ctx.save_for_backward(*inputs)
            ctx.func = func
            
            # 前向计算（不追踪梯度）
            with torch.no_grad():
                outputs = func(*inputs)
            
            return outputs
        
        @staticmethod
        def backward(ctx, *grad_outputs):
            inputs = ctx.saved_tensors
            
            # 重新启用梯度计算
            with torch.enable_grad():
                # 创建需要梯度的输入副本
                inputs_with_grad = [inp.detach().requires_grad_(True) for inp in inputs]
                
                # 重新计算前向
                outputs = ctx.func(*inputs_with_grad)
                
                # 计算梯度
                torch.autograd.backward(outputs, grad_outputs)
            
            return tuple(inp.grad for inp in inputs_with_grad)
    
    return CheckpointFunction.apply(*args)


# ==================== 内存分析 ====================
def analyze_memory_savings(num_layers: int, hidden_size: int, seq_len: int, batch_size: int):
    """分析 Checkpointing 的内存节省"""
    
    # 估算每层的激活内存（简化）
    # 主要是 attention scores [batch, heads, seq, seq] 和 hidden states [batch, seq, hidden]
    activation_per_layer = (
        batch_size * seq_len * hidden_size * 2 +  # FFN 中间激活
        batch_size * seq_len * seq_len * 2         # Attention scores (假设 2 bytes per value)
    )
    
    standard_memory = num_layers * activation_per_layer
    checkpoint_memory = math.sqrt(num_layers) * activation_per_layer  # 只保存 √n 个检查点
    
    print(f"=== Gradient Checkpointing 内存分析 ===")
    print(f"配置: {num_layers} layers, hidden={hidden_size}, seq={seq_len}, batch={batch_size}")
    print(f"标准训练内存: {standard_memory / 1024**3:.2f} GB")
    print(f"Checkpointing 内存: {checkpoint_memory / 1024**3:.2f} GB")
    print(f"节省: {(1 - checkpoint_memory/standard_memory) * 100:.1f}%")


if __name__ == "__main__":
    analyze_memory_savings(
        num_layers=32,
        hidden_size=4096,
        seq_len=2048,
        batch_size=4
    )
```

---

## Mixed Precision Training

### 🎯 核心思想

使用 FP16/BF16 加速计算，用 FP32 保证精度：
- **前向/反向**: FP16（快、省内存）
- **参数更新**: FP32（保证精度）
- **Loss Scaling**: 防止 FP16 梯度下溢

### 📝 实现代码

```python
class MixedPrecisionTrainer:
    """
    混合精度训练
    
    核心组件：
    1. GradScaler: 梯度缩放，防止 FP16 下溢
    2. autocast: 自动选择 FP16/FP32
    """
    
    def __init__(self, model, optimizer, init_scale=65536.0):
        self.model = model
        self.optimizer = optimizer
        
        # 梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
    
    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        
        # 使用 autocast 自动混合精度
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
        
        # 反向传播（使用缩放后的 loss）
        self.scaler.scale(loss).backward()
        
        # 先 unscale 梯度，然后裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数（scaler 会检查是否有 inf/nan）
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


class GradScaler:
    """
    手动实现 Gradient Scaler
    
    为什么需要？
    - FP16 的最小正数是 ~6e-8
    - 深度网络的梯度可能小于这个值，导致下溢为 0
    - 缩放 loss 使梯度变大，更新时再缩放回来
    """
    
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        
        self.step_count = 0
        self.inf_count = 0
    
    def scale_loss(self, loss):
        """缩放 loss"""
        return loss * self.scale
    
    def unscale_grads(self, optimizer):
        """反缩放梯度"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def step(self, optimizer):
        """
        执行优化器 step，并处理 inf/nan
        """
        # 检查是否有 inf/nan 梯度
        has_inf = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        has_inf = True
                        break
        
        if has_inf:
            # 跳过这次更新，减小 scale
            self.inf_count += 1
            self.scale *= self.backoff_factor
            print(f"发现 inf/nan，scale 降低到 {self.scale}")
            return False
        else:
            # 正常更新
            optimizer.step()
            self.step_count += 1
            
            # 定期增大 scale
            if self.step_count % self.growth_interval == 0:
                self.scale *= self.growth_factor
            
            return True
    
    def update(self):
        """更新内部状态"""
        pass  # 在 step 中已处理
```

---

## Gradient Accumulation

### 🎯 核心思想

用小 batch 模拟大 batch：累积多个 mini-batch 的梯度后再更新。

**等效 batch size** = mini_batch_size × accumulation_steps

### 📝 实现代码

```python
class GradientAccumulationTrainer:
    """
    梯度累积训练
    
    用小 batch 模拟大 batch
    """
    
    def __init__(
        self,
        model,
        optimizer,
        accumulation_steps: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def train_step(self, inputs, labels):
        """
        累积梯度的训练步骤
        """
        # 计算 loss（除以累积步数，保证梯度量级正确）
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss = loss / self.accumulation_steps
        
        # 反向传播（梯度累积）
        loss.backward()
        
        self.step_count += 1
        
        # 每 accumulation_steps 步更新一次
        if self.step_count % self.accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return True  # 发生了更新
        
        return False  # 只累积，未更新
    
    def train_epoch(self, dataloader):
        """
        完整的一个 epoch
        """
        self.model.train()
        total_loss = 0
        update_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels) / self.accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.accumulation_steps
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                update_count += 1
        
        # 处理最后不足 accumulation_steps 的 batch
        if (batch_idx + 1) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / len(dataloader)
```

---

## 面试追问汇总

### LoRA 相关

| 问题 | 答案 |
|:---|:---|
| LoRA 的参数量比例 | (d+k)r / dk，r=8 时约 0.4% |
| 为什么 B 初始化为 0 | 保证从原模型开始，不破坏预训练 |
| rank 选多大 | 通常 8-64，任务越复杂 rank 越大 |
| 哪些层应用 LoRA | 通常 Q, V 投影效果最好 |

### Checkpointing 相关

| 问题 | 答案 |
|:---|:---|
| 时间换空间的代价 | 约增加 30% 训练时间 |
| 内存节省多少 | O(n) → O(√n) |
| 什么时候用 | 显存不够时 |

### 混合精度相关

| 问题 | 答案 |
|:---|:---|
| 为什么需要 Loss Scaling | FP16 梯度可能下溢 |
| BF16 vs FP16 | BF16 范围更大，不需要 scaling |
| 哪些操作用 FP32 | Softmax, LayerNorm, Loss |

---

## 🔗 相关题目

- [Adam 优化器](06-optimizers.md#adam) - LoRA 训练常用
- [KV Cache](09-inference-optimization.md#kv-cache) - 推理优化
