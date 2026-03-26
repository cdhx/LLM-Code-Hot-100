# LLM 基础 - 深度学习基石

> 万丈高楼平地起，这些基础概念是理解 LLM 的根基

## 📑 目录

- [方法一览对比](#方法一览对比)
- [梯度与反向传播](#梯度与反向传播)
- [线性回归](#线性回归)
- [逻辑回归](#逻辑回归)
- [MLP 多层感知机](#mlp-多层感知机)
- [激活函数](#激活函数)

---

## 方法一览对比

### 回归 vs 分类

| 方法 | 任务类型 | 输出 | 损失函数 | 核心区别 |
|:---:|:---:|:---:|:---:|:---|
| **Linear Regression** | 回归 | 连续值 | MSE | 直接输出 `Wx + b` |
| **Logistic Regression** | 二分类 | 概率 [0,1] | BCE | 加 `sigmoid(Wx + b)` |
| **Softmax Regression** | 多分类 | 概率分布 | CE | 加 `softmax(Wx + b)` |

### 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 |
|:---:|:---:|:---|:---|
| **Sigmoid** | `1/(1+e^(-x))` | 输出 (0,1)，可解释为概率 | 梯度消失，非零中心 |
| **Tanh** | `(e^x-e^(-x))/(e^x+e^(-x))` | 零中心 | 梯度消失 |
| **ReLU** | `max(0, x)` | 计算简单，缓解梯度消失 | Dead ReLU（负数梯度为0） |
| **GELU** | `x * Φ(x)` | 平滑，LLM 主流 | 计算稍复杂 |
| **SiLU/Swish** | `x * sigmoid(x)` | 平滑，性能好 | 计算稍复杂 |

---

## 梯度与反向传播

### 🎯 核心思想

反向传播 = **链式法则** 的系统应用，从输出层向输入层逐层计算梯度

### 📝 手动实现反向传播

```python
import numpy as np

class ComputeGraph:
    """计算图：手动实现前向传播和反向传播"""
    
    def __init__(self):
        self.grad = {}  # 存储各变量的梯度
    
    def forward(self, x, W1, b1, W2, b2):
        """前向传播: x -> Linear -> ReLU -> Linear -> output"""
        # 第一层线性变换
        self.x = x
        self.W1, self.b1 = W1, b1
        self.z1 = x @ W1 + b1           # [batch, hidden]
        
        # ReLU 激活
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # 第二层线性变换
        self.W2, self.b2 = W2, b2
        self.z2 = self.a1 @ W2 + b2      # [batch, output]
        
        return self.z2
    
    def backward(self, grad_output):
        """反向传播：从输出梯度计算各参数梯度
        
        链式法则: dL/dW = dL/dz * dz/dW
        """
        batch_size = self.x.shape[0]
        
        # ===== 第二层梯度 =====
        # dL/dz2 = grad_output (来自损失函数)
        dz2 = grad_output
        
        # dL/dW2 = a1^T @ dz2
        self.grad['W2'] = self.a1.T @ dz2 / batch_size
        # dL/db2 = sum(dz2)
        self.grad['b2'] = dz2.mean(axis=0)
        
        # dL/da1 = dz2 @ W2^T (传递给上一层)
        da1 = dz2 @ self.W2.T
        
        # ===== ReLU 梯度 =====
        # dL/dz1 = dL/da1 * da1/dz1
        # ReLU 梯度: z1 > 0 则为 1，否则为 0
        dz1 = da1 * (self.z1 > 0).astype(float)
        
        # ===== 第一层梯度 =====
        # dL/dW1 = x^T @ dz1
        self.grad['W1'] = self.x.T @ dz1 / batch_size
        # dL/db1 = sum(dz1)
        self.grad['b1'] = dz1.mean(axis=0)
        
        # dL/dx = dz1 @ W1^T (如果需要继续传递)
        dx = dz1 @ self.W1.T
        
        return dx

# 使用示例
np.random.seed(42)
x = np.random.randn(4, 3)      # batch=4, input_dim=3
W1 = np.random.randn(3, 5)     # hidden_dim=5
b1 = np.zeros(5)
W2 = np.random.randn(5, 2)     # output_dim=2
b2 = np.zeros(2)

graph = ComputeGraph()
output = graph.forward(x, W1, b1, W2, b2)
grad_output = np.ones_like(output)  # 假设损失对输出的梯度
graph.backward(grad_output)

print(f"dW1 shape: {graph.grad['W1'].shape}")  # (3, 5)
print(f"dW2 shape: {graph.grad['W2'].shape}")  # (5, 2)
```

🤔 **Q: 为什么要除以 batch_size？**

> 这是为了计算**平均梯度**。如果不除，梯度会随 batch size 变大而变大，学习率就需要相应调整。除以 batch_size 让梯度与 batch 大小解耦，学习率更稳定。

🤔 **Q: ReLU 在 x=0 处不可导，怎么办？**

> 实践中直接定义 x=0 处的梯度为 0（或 1）。因为 x 精确等于 0 的概率极低，不影响训练。

### 📝 PyTorch Autograd 验证

```python
import torch
import torch.nn as nn

# 用 PyTorch 验证手动实现的正确性
x_pt = torch.tensor(x, requires_grad=True, dtype=torch.float32)
W1_pt = torch.tensor(W1, requires_grad=True, dtype=torch.float32)
b1_pt = torch.tensor(b1, requires_grad=True, dtype=torch.float32)
W2_pt = torch.tensor(W2, requires_grad=True, dtype=torch.float32)
b2_pt = torch.tensor(b2, requires_grad=True, dtype=torch.float32)

# 前向传播
z1 = x_pt @ W1_pt + b1_pt
a1 = torch.relu(z1)
z2 = a1 @ W2_pt + b2_pt
loss = z2.sum()

# 反向传播
loss.backward()

print(f"PyTorch dW1:\n{W1_pt.grad / 4}")  # 除以4因为我们用的是sum而不是mean
print(f"Manual dW1:\n{graph.grad['W1']}")
# 两者应该一致！
```

---

## 线性回归

### 🎯 核心思想

找到最佳的 W 和 b，使得 `y_pred = Wx + b` 尽可能接近真实值 y

### 📝 实现代码

```python
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    """线性回归：最基础的回归模型"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)  # 直接输出 Wx + b

# 从零实现
class LinearRegressionManual:
    """手动实现线性回归（含梯度下降）"""
    
    def __init__(self, input_dim, output_dim, lr=0.01):
        # 参数初始化
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.lr = lr
    
    def forward(self, x):
        """前向传播: y = Wx + b"""
        return x @ self.W + self.b
    
    def mse_loss(self, y_pred, y_true):
        """均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, x, y_pred, y_true):
        """计算梯度并更新参数
        
        Loss = mean((y_pred - y_true)^2)
        dL/dy_pred = 2 * (y_pred - y_true) / n
        dL/dW = x^T @ dL/dy_pred
        dL/db = sum(dL/dy_pred)
        """
        batch_size = x.shape[0]
        grad_output = 2 * (y_pred - y_true) / batch_size
        
        # 梯度计算
        grad_W = x.T @ grad_output / batch_size
        grad_b = grad_output.mean(axis=0)
        
        # 梯度下降更新
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
    
    def fit(self, x, y, epochs=100):
        """训练"""
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.mse_loss(y_pred, y)
            self.backward(x, y_pred, y)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 使用示例
np.random.seed(42)
x = np.random.randn(100, 3)
y = x @ np.array([[1], [2], [3]]) + 0.5 + np.random.randn(100, 1) * 0.1

model = LinearRegressionManual(3, 1, lr=0.1)
model.fit(x, y, epochs=100)
print(f"Learned W: {model.W.flatten()}")  # 应该接近 [1, 2, 3]
print(f"Learned b: {model.b}")            # 应该接近 0.5
```

---

## 逻辑回归

### 🎯 核心思想

线性回归 + Sigmoid = 逻辑回归，将输出压缩到 (0, 1) 作为概率

### 📝 实现代码

```python
class LogisticRegression(nn.Module):
    """逻辑回归：线性模型 + Sigmoid，用于二分类"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # 输出概率
    
    def predict(self, x, threshold=0.5):
        """预测类别"""
        return (self.forward(x) > threshold).float()

# 从零实现
class LogisticRegressionManual:
    """手动实现逻辑回归"""
    
    def __init__(self, input_dim, lr=0.01):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros(1)
        self.lr = lr
    
    def sigmoid(self, z):
        """Sigmoid 函数，注意数值稳定性"""
        # 避免 exp 溢出
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def forward(self, x):
        """前向传播: p = sigmoid(Wx + b)"""
        z = x @ self.W + self.b
        return self.sigmoid(z)
    
    def bce_loss(self, y_pred, y_true):
        """Binary Cross Entropy Loss
        
        Loss = -[y*log(p) + (1-y)*log(1-p)]
        """
        eps = 1e-7  # 防止 log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, x, y_pred, y_true):
        """计算梯度并更新
        
        神奇的是，BCE + Sigmoid 的梯度形式很简洁:
        dL/dz = y_pred - y_true
        """
        batch_size = x.shape[0]
        grad_z = (y_pred - y_true) / batch_size
        
        # 梯度计算
        grad_W = x.T @ grad_z
        grad_b = grad_z.mean()
        
        # 更新
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
    
    def fit(self, x, y, epochs=100):
        """训练"""
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.bce_loss(y_pred, y)
            self.backward(x, y_pred, y)
            if epoch % 20 == 0:
                acc = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}")

# 使用示例
np.random.seed(42)
x = np.random.randn(200, 2)
y = ((x[:, 0] + x[:, 1]) > 0).reshape(-1, 1).astype(float)  # 简单的线性可分数据

model = LogisticRegressionManual(2, lr=0.5)
model.fit(x, y, epochs=100)
```

🤔 **Q: 为什么 BCE + Sigmoid 的梯度这么简洁 (y_pred - y_true)？**

> 这不是巧合！BCE Loss 正是为了配合 Sigmoid 设计的。数学上：
> - `dBCE/dp = -y/p + (1-y)/(1-p)`
> - `dsigmoid/dz = p(1-p)`
> - 链式法则：`dBCE/dz = dBCE/dp * dp/dz = p - y`
> 
> 这个优美的形式让计算非常高效。

🤔 **Q: 为什么 Sigmoid 要分段计算？**

> 当 z 是很大的负数时，`exp(-z)` 会溢出到 inf。改用 `exp(z)/(1+exp(z))` 在 z < 0 时更稳定。

---

## MLP 多层感知机

### 🎯 核心思想

多个线性层 + 非线性激活堆叠，可以拟合任意复杂函数（万能近似定理）

### 📝 实现代码

```python
class MLP(nn.Module):
    """多层感知机：通用的前馈神经网络"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 从零实现（含反向传播）
class MLPManual:
    """手动实现 MLP"""
    
    def __init__(self, layer_dims, lr=0.01):
        """
        layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.num_layers = len(layer_dims) - 1
        self.lr = lr
        
        # 初始化参数（Xavier 初始化）
        self.params = {}
        for i in range(self.num_layers):
            scale = np.sqrt(2.0 / layer_dims[i])
            self.params[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i+1]) * scale
            self.params[f'b{i}'] = np.zeros(layer_dims[i+1])
        
        # 缓存中间结果用于反向传播
        self.cache = {}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """前向传播"""
        self.cache['a0'] = x
        
        for i in range(self.num_layers):
            z = self.cache[f'a{i}'] @ self.params[f'W{i}'] + self.params[f'b{i}']
            self.cache[f'z{i}'] = z
            
            # 最后一层不加激活（或根据任务选择）
            if i < self.num_layers - 1:
                self.cache[f'a{i+1}'] = self.relu(z)
            else:
                self.cache[f'a{i+1}'] = z  # 输出层
        
        return self.cache[f'a{self.num_layers}']
    
    def backward(self, grad_output):
        """反向传播"""
        batch_size = self.cache['a0'].shape[0]
        grads = {}
        
        da = grad_output
        
        for i in range(self.num_layers - 1, -1, -1):
            # 如果不是输出层，需要经过 ReLU 梯度
            if i < self.num_layers - 1:
                dz = da * self.relu_grad(self.cache[f'z{i}'])
            else:
                dz = da
            
            # 计算参数梯度
            grads[f'W{i}'] = self.cache[f'a{i}'].T @ dz / batch_size
            grads[f'b{i}'] = dz.mean(axis=0)
            
            # 传递给上一层
            da = dz @ self.params[f'W{i}'].T
        
        # 更新参数
        for i in range(self.num_layers):
            self.params[f'W{i}'] -= self.lr * grads[f'W{i}']
            self.params[f'b{i}'] -= self.lr * grads[f'b{i}']
        
        return grads
    
    def fit(self, x, y, epochs=100):
        """训练（MSE Loss）"""
        for epoch in range(epochs):
            # 前向
            y_pred = self.forward(x)
            
            # MSE Loss
            loss = np.mean((y_pred - y) ** 2)
            
            # 反向
            grad_output = 2 * (y_pred - y) / y.shape[0]
            self.backward(grad_output)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 使用示例：拟合 sin 函数
np.random.seed(42)
x = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y = np.sin(x)

model = MLPManual([1, 32, 32, 1], lr=0.01)
model.fit(x, y, epochs=1000)
```

🤔 **Q: 为什么要用 Xavier 初始化？**

> 如果初始化太大，激活值会饱和，梯度消失；太小，信号会逐层衰减。Xavier 初始化使得每层的输入和输出方差一致，训练更稳定。对于 ReLU，更推荐 He 初始化：`scale = sqrt(2/n)`。

🤔 **Q: 为什么输出层不加激活函数？**

> 取决于任务：
> - **回归任务**：不加激活，直接输出
> - **二分类**：加 Sigmoid
> - **多分类**：加 Softmax
> - **LLM 输出层**：不加激活，后面接 Softmax（在 loss 里计算）

---

## 激活函数

### 📝 实现代码

```python
class Activations:
    """常用激活函数及其梯度"""
    
    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    @staticmethod
    def sigmoid_grad(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_grad(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_grad(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def gelu(x):
        """GELU: Gaussian Error Linear Unit (LLM 常用)"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def silu(x):
        """SiLU/Swish: x * sigmoid(x)"""
        return x * Activations.sigmoid(x)

# PyTorch 版本
class ActivationsPyTorch:
    """PyTorch 中的激活函数"""
    
    @staticmethod
    def demo():
        x = torch.randn(5)
        
        print("ReLU:", torch.relu(x))
        print("GELU:", torch.nn.functional.gelu(x))
        print("SiLU:", torch.nn.functional.silu(x))
        print("Sigmoid:", torch.sigmoid(x))
        print("Tanh:", torch.tanh(x))
```

---

## 💡 面试常见追问

### 1. 梯度消失和梯度爆炸

**Q: 什么是梯度消失/爆炸？如何解决？**

梯度消失：深层网络中，梯度逐层相乘变得很小，底层参数几乎不更新
梯度爆炸：梯度逐层相乘变得很大，参数更新剧烈导致不稳定

**解决方案：**
- 激活函数：用 ReLU 替代 Sigmoid/Tanh
- 归一化：BatchNorm, LayerNorm
- 残差连接：ResNet 的 skip connection
- 梯度裁剪：`torch.nn.utils.clip_grad_norm_`
- 合适的初始化：Xavier, He

### 2. 为什么需要非线性激活？

没有激活函数，多层线性变换等价于一层：`W2(W1x) = (W2W1)x = Wx`
非线性激活打破了这个限制，让网络能拟合复杂函数

### 3. Softmax 的数值稳定性

```python
def softmax_stable(x):
    """数值稳定的 Softmax"""
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)  # 减去最大值防止溢出
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

---

## 🔗 相关题目

- [05-loss-functions.md](05-loss-functions.md) - Cross Entropy, MSE 等损失函数的详细实现
- [06-optimizers.md](06-optimizers.md) - SGD, Adam 等优化器的详细实现
- [10-transformer-architecture.md](10-transformer-architecture.md) - FFN 层使用了这里的 MLP 结构
