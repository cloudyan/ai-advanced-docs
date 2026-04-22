# 第 4 章：深度学习基础（6-8 周）

> 神经网络的奥秘 —— 从感知机到 CNN/RNN
> 
> _学习周期：6-8 周 | 难度：⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 深度学习 vs 机器学习

```
机器学习：
  特征工程 + 分类器
  ┌──────────┐    ┌──────────┐
  │ 人工特征 │───▶│ 分类器   │
  └──────────┘    └──────────┘

深度学习：
  端到端学习
  ┌──────────────────────────┐
  │ 原始输入 → 多层神经网络 → 输出 │
  └──────────────────────────┘
        自动学习特征表示
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 理解神经网络的前向传播和反向传播
- ✅ 掌握常见激活函数的特点和选择
- ✅ 实现 CNN 进行图像分类
- ✅ 实现 RNN/LSTM 处理序列数据
- ✅ 使用 PyTorch 构建完整深度学习模型

---

## 📚 学习大纲

### 4.1 神经网络基础（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 感知机（Perceptron）

```python
import numpy as np

class Perceptron:
    """感知机：最简单的神经网络"""
    
    def __init__(self, input_size, lr=0.01):
        # 初始化权重和偏置
        self.weights = np.random.randn(input_size)
        self.bias = 0
        self.lr = lr
    
    def step_function(self, x):
        """阶跃激活函数"""
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        """前向传播"""
        linear = np.dot(X, self.weights) + self.bias
        return self.step_function(linear)
    
    def fit(self, X, y, epochs=100):
        """训练"""
        for epoch in range(epochs):
            for i in range(len(X)):
                pred = self.predict(X[i])
                error = y[i] - pred
                
                # 权重更新
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

# 测试：实现 AND 门
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND

perceptron = Perceptron(input_size=2, lr=0.1)
perceptron.fit(X, y, epochs=100)

print("AND 门预测结果：")
for i in range(len(X)):
    print(f"{X[i]} → {perceptron.predict(X[i])}")
```

#### 多层感知机（MLP）

```python
import numpy as np

class MLP:
    """两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        """ReLU 激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU 导数"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax 激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向传播"""
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output, lr=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差
        delta2 = output - y  # (m, output_size)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # 更新权重
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def train(self, X, y, epochs=1000, lr=0.01):
        """训练循环"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失（交叉熵）
            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            
            # 反向传播
            self.backward(X, y, output, lr)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 测试：XOR 问题（单层感知机无法解决）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot 编码

mlp = MLP(input_size=2, hidden_size=4, output_size=2)
mlp.train(X, y, epochs=1000, lr=0.1)

print("\nXOR 问题预测结果：")
predictions = mlp.predict(X)
for i in range(len(X)):
    print(f"{X[i]} → {predictions[i]}")
```

#### 激活函数对比

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# 可视化
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.plot(x, gelu(x))
plt.title('GELU')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 对比表
print("""
激活函数对比：
┌─────────────┬──────────────┬─────────────┬─────────────┐
│ 函数        │ 输出范围     │ 优点        │ 缺点        │
├─────────────┼──────────────┼─────────────┼─────────────┤
│ Sigmoid     │ (0, 1)       │ 平滑        │ 梯度消失    │
│ Tanh        │ (-1, 1)      │ 零中心化    │ 梯度消失    │
│ ReLU        │ [0, ∞)       │ 计算快      │ 神经元死亡  │
│ Leaky ReLU  │ (-∞, ∞)      │ 解决死亡    │ 需要调 alpha │
│ GELU        │ (-∞, ∞)      │ 平滑        │ 计算复杂    │
└─────────────┴──────────────┴─────────────┴─────────────┘

推荐：
- 隐藏层：ReLU（默认）、GELU（Transformer）
- 输出层（二分类）：Sigmoid
- 输出层（多分类）：Softmax
""")
```

#### 反向传播详解

```python
import numpy as np

def backward_propagation_demo():
    """
    反向传播手动推导示例
    
    网络结构：
    输入 (2) → 隐藏层 (3) → 输出 (2)
    
    损失函数：交叉熵
    """
    
    # 前向传播结果
    X = np.array([[0.5, -0.2]])      # 输入
    z1 = np.array([[0.3, -0.5, 0.8]]) # 隐藏层输入
    a1 = np.array([[0.57, 0.38, 0.69]]) # 隐藏层输出（ReLU 后）
    z2 = np.array([[0.4, -0.3]])      # 输出层输入
    a2 = np.array([[0.6, 0.4]])       # 输出（Softmax 后）
    
    # 真实标签
    y = np.array([[1, 0]])
    
    # 权重
    W1 = np.random.randn(2, 3)
    W2 = np.random.randn(3, 2)
    
    print("=== 反向传播逐步计算 ===\n")
    
    # 步骤 1：输出层误差
    # L = -Σy·log(a2)
    # dL/da2 = -y/a2
    # da2/dz2 = a2*(1-a2) (Softmax 导数)
    # dL/dz2 = a2 - y (简化后)
    delta2 = a2 - y
    print(f"1. 输出层误差 δ2 = a2 - y = {delta2}")
    
    # 步骤 2：计算 W2 的梯度
    # dL/dW2 = a1.T @ delta2
    dW2 = np.dot(a1.T, delta2)
    print(f"2. W2 梯度 dW2 = a1.T @ δ2 = {dW2}")
    
    # 步骤 3：隐藏层误差
    # dL/da1 = delta2 @ W2.T
    # da1/dz1 = ReLU 导数
    # δ1 = (delta2 @ W2.T) * ReLU'(z1)
    delta1 = np.dot(delta2, W2.T) * (z1 > 0)
    print(f"3. 隐藏层误差 δ1 = (δ2 @ W2.T) * ReLU'(z1)")
    
    # 步骤 4：计算 W1 的梯度
    # dL/dW1 = X.T @ delta1
    dW1 = np.dot(X.T, delta1)
    print(f"4. W1 梯度 dW1 = X.T @ δ1 = {dW1}")
    
    print("\n=== 反向传播完成 ===")
    print("梯度用于更新权重：W = W - lr * dW")

backward_propagation_demo()
```

</details>

---

### 4.2 卷积神经网络 CNN（2-3 周）

<details>
<summary>📋 查看详细知识点</summary>

#### CNN 核心组件

```
CNN 架构：
输入图片 → 卷积层 → 激活 → 池化 → 卷积 → 激活 → 池化 → 全连接 → 输出

关键组件：
1. 卷积层：提取局部特征
2. 池化层：降维、不变性
3. 全连接层：分类
```

#### 卷积层详解

```python
import numpy as np

def conv2d(input_image, kernel, stride=1, padding=0):
    """
    2D 卷积实现
    
    参数：
    - input_image: (H, W) 或 (H, W, C)
    - kernel: (kH, kW) 卷积核
    - stride: 步长
    - padding: 填充
    """
    # 添加 padding
    if padding > 0:
        input_image = np.pad(input_image, padding, mode='constant')
    
    h, w = input_image.shape[:2]
    kh, kw = kernel.shape[:2]
    
    # 计算输出尺寸
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    # 输出
    output = np.zeros((out_h, out_w))
    
    # 卷积操作
    for i in range(out_h):
        for j in range(out_w):
            region = input_image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# 常见卷积核
kernels = {
    '边缘检测': np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]),
    
    '锐化': np.array([[ 0, -1,  0],
                      [-1,  5, -1],
                      [ 0, -1,  0]]),
    
    '模糊': np.array([[1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9]]),
    
    'Sobel X': np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]),
    
    'Sobel Y': np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
}

# 测试
image = np.random.randn(28, 28)
for name, kernel in kernels.items():
    result = conv2d(image, kernel)
    print(f"{name}卷积核输出形状：{result.shape}")
```

#### PyTorch 实现 CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 架构（经典 CNN）
    
    输入：32x32 灰度图
    输出：10 类别
    """
    def __init__(self):
        super().__init__()
        
        # 卷积层 1: 1→6, 5x5 卷积
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 输出：6x28x28 → 池化 → 6x14x14
        
        # 卷积层 2: 6→16, 5x5 卷积
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 输出：16x10x10 → 池化 → 16x5x5
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # 卷积块 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积块 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# 测试
model = LeNet5()
x = torch.randn(32, 1, 32, 32)  # batch=32, 1 通道，32x32
output = model(x)
print(f"输出形状：{output.shape}")  # (32, 10)

# 参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数：{total_params:,}")
print(f"可训练参数：{trainable_params:,}")
```

#### 经典 CNN 架构对比

```python
"""
CNN 架构演进：

1. LeNet-5 (1998)
   - 2 个卷积层 + 3 个全连接层
   - 参数：~60K
   - 应用：手写数字识别

2. AlexNet (2012) ⭐
   - 5 个卷积层 + 3 个全连接层
   - 创新：ReLU、Dropout、数据增强
   - 参数：~60M
   - ImageNet 冠军（15.3% top-5 错误率）

3. VGG (2014)
   - 使用小卷积核 (3x3)
   - VGG-16: 16 层，138M 参数
   - 特点：结构规整，特征提取强

4. ResNet (2015) ⭐
   - 残差连接：y = F(x) + x
   - 可以训练非常深的网络（152 层+）
   - ImageNet 冠军（3.57% top-5 错误率）

5. DenseNet (2017)
   - 密集连接：每层连接所有前面层
   - 参数效率高
   - 梯度流动更好
"""

import torch.nn as nn

class ResidualBlock(nn.Module):
    """ResNet 残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 短路连接（处理维度不匹配）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

# ResNet-18 简化版
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

</details>

---

### 4.3 循环神经网络 RNN（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### RNN 基础

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """简单 RNN 实现"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # RNN 单元
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        output, hidden = self.rnn(x, hidden)
        # output: (batch, seq_len, hidden_size)
        
        # 取最后一个时间步的输出
        output = self.fc(output[:, -1, :])
        return output, hidden

# 测试
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(32, 15, 10)  # batch=32, seq_len=15, input=10
output, hidden = rnn(x)
print(f"输出形状：{output.shape}")  # (32, 5)
```

#### LSTM 详解

```python
class LSTMCell:
    """
    LSTM 单元手动实现
    
    核心公式：
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # 遗忘门
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # 输入门
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # 候选记忆
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # 输出门
    
    c_t = f_t * c_{t-1} + i_t * g_t  # 更新细胞状态
    h_t = o_t * tanh(c_t)  # 更新隐藏状态
    """
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # 门权重（4 个门：遗忘、输入、候选、输出）
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden, cell):
        # 拼接输入和隐藏状态
        combined = torch.cat((hidden, x), dim=1)
        
        # 计算四个门
        f = self.sigmoid(self.W_f(combined))  # 遗忘门
        i = self.sigmoid(self.W_i(combined))  # 输入门
        g = self.tanh(self.W_g(combined))     # 候选记忆
        o = self.sigmoid(self.W_o(combined))  # 输出门
        
        # 更新状态
        cell = f * cell + i * g
        hidden = o * self.tanh(cell)
        
        return hidden, cell

# PyTorch 内置 LSTM
class LSTMClassifier(nn.Module):
    """LSTM 文本分类器"""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向×2
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        # embedded: (batch, seq_len, embed_size)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden: (num_layers*2, batch, hidden_size)
        
        # 拼接最后一层的双向隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return output

# 测试
vocab_size = 10000
model = LSTMClassifier(vocab_size, embed_size=300, hidden_size=256, num_classes=10)
x = torch.randint(0, vocab_size, (32, 50))  # batch=32, seq_len=50
output = model(x)
print(f"输出形状：{output.shape}")  # (32, 10)
```

#### GRU（简化版 LSTM）

```python
"""
GRU vs LSTM:

LSTM:
- 3 个门：遗忘门、输入门、输出门
- 2 个状态：细胞状态 c_t、隐藏状态 h_t

GRU:
- 2 个门：更新门、重置门
- 1 个状态：隐藏状态 h_t
- 参数更少，训练更快

选择：
- 数据量大：LSTM（表达能力强）
- 数据量小：GRU（不易过拟合）
- 资源受限：GRU（参数少）
"""

class GRUClassifier(nn.Module):
    """GRU 文本分类器"""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, 
                         bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        # hidden: (num_layers*2, batch, hidden_size)
        
        # 拼接双向隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output
```

#### Seq2Seq 模型

```python
class Encoder(nn.Module):
    """Seq2Seq 编码器"""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    """Seq2Seq 解码器"""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x.unsqueeze(1))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """完整 Seq2Seq 模型"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码
        hidden, cell = self.encoder(src)
        
        # 解码器输入（起始符）
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            
            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        return outputs
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 4.1 神经网络基础 | 2 周 | - | 0% | ⏳ |
| 4.2 CNN | 2-3 周 | - | 0% | ⏳ |
| 4.3 RNN/LSTM | 2 周 | - | 0% | ⏳ |

### 项目清单

- [ ] 从零实现 MLP 和反向传播
- [ ] CIFAR-10 图像分类（CNN）
- [ ] ResNet 迁移学习
- [ ] IMDB 情感分析（LSTM）
- [ ] 简单翻译模型（Seq2Seq）

---

> _深度学习是神经网络的复兴，卷积看世界，循环记时光。_
> 
> _—— 悟空_
