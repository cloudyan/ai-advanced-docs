# 5.1 注意力机制详解

> Transformer 的灵魂 —— 深入理解自注意力机制
> 
> _预计时间：1-2 周 | 难度：⭐⭐⭐⭐ | 前置知识：矩阵运算、softmax_

---

## 📖 内容概览

```
注意力机制学习路径：

Day 1-2: 注意力基本概念
Day 3-4: 自注意力机制
Day 5-6: 多头注意力
Day 7: 注意力可视化与分析
```

---

## 1. 注意力基本概念

### 什么是注意力？

```
人类注意力：
当你看一张照片时，不会平等地看待每个像素
而是聚焦于关键区域（人脸、文字等）
忽略背景和不重要的部分

机器注意力：
当模型处理一句话时，不是平等对待每个词
而是给重要的词分配更多"注意力"
从而更好地理解和生成
```

### 注意力分数计算

```python
import torch
import torch.nn.functional as F

"""
注意力机制核心公式：

Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中：
- Q (Query): 查询向量，"我要找什么"
- K (Key):   键向量，"我有什么"
- V (Value): 值向量，"具体内容"
- d_k:       Q 和 K 的维度，用于缩放
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力实现
    
    参数：
    - Q, K, V: (batch_size, seq_len, d_k)
    - mask: 可选的掩码矩阵
    
    返回：
    - output: 注意力输出
    - attention_weights: 注意力权重（用于可视化）
    """
    d_k = Q.size(-1)
    
    # 1. 计算注意力分数 Q·K^T
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) = (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. 缩放（防止点积过大导致 softmax 梯度消失）
    scores = scores / (d_k ** 0.5)
    
    # 3. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和 V
    # (batch, seq_len, seq_len) @ (batch, seq_len, d_k) = (batch, seq_len, d_k)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# 测试
batch_size = 2
seq_len = 4
d_k = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"输出形状：{output.shape}")  # (2, 4, 8)
print(f"注意力权重形状：{attn_weights.shape}")  # (2, 4, 4)
```

---

## 2. 自注意力机制（Self-Attention）

### 自注意力原理

```
自注意力 vs 普通注意力：

普通注意力：
- Q 来自一个序列（如解码器）
- K, V 来自另一个序列（如编码器）
- 用于 Encoder-Decoder 之间的注意力

自注意力：
- Q, K, V 都来自同一个序列
- 用于序列内部的关系建模
- 每个位置都能关注到所有位置
```

### 自注意力实现

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """自注意力层"""
    
    def __init__(self, embed_size, heads):
        """
        参数：
        - embed_size: 词嵌入维度
        - heads: 注意力头数（多头时用）
        """
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert embed_size % heads == 0, "embed_size 必须能被 heads 整除"
        
        # 学习 Q, K, V 的线性投影
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 1. 线性投影
        queries = self.query(Q)
        keys = self.key(K)
        values = self.value(V)
        
        # 2. 缩放点积注意力
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        
        # 3. 加权求和
        out = torch.matmul(attention, values)
        
        # 4. 输出投影
        out = self.fc_out(out)
        
        return out, attention

# 测试
embed_size = 512
heads = 8
self_attn = SelfAttention(embed_size, heads)

# 输入：batch=4, seq_len=100, embed_size=512
Q = K = V = torch.randn(4, 100, 512)
output, attn_weights = self_attn(Q, K, V)
print(f"输出形状：{output.shape}")  # (4, 100, 512)
```

### 自注意力的几何解释

```python
"""
自注意力的几何意义：

1. 对于序列中的每个位置 i：
   - 计算它的 Query 向量 Q_i
   - 与所有位置的 Key 向量计算相似度
   - 得到注意力权重分布

2. 注意力权重表示：
   - 位置 i 应该"关注"哪些位置
   - 权重越高，表示关系越密切

3. 加权求和：
   - 根据注意力权重聚合 Value
   - 得到包含上下文信息的新表示

示例：
句子："我 爱 中国"

对于"爱"这个位置：
- 与"我"的注意力权重：0.3
- 与"爱"的注意力权重：0.5（自身通常较高）
- 与"中国"的注意力权重：0.2

输出 = 0.3×V_我 + 0.5×V_爱 + 0.2×V_中国
"""
```

---

## 3. 多头注意力（Multi-Head Attention）

### 为什么需要多头？

```
单头注意力的局限：
- 只能学习一种注意力模式
- 可能遗漏不同类型的关系

多头的优势：
- 多个头并行学习不同的表示子空间
- 每个头可以关注不同类型的关系

例如：
Head 1: 关注语法关系（主谓宾）
Head 2: 关注语义关系（同义词）
Head 3: 关注指代关系（代词 - 名词）
Head 4: 关注位置关系（相邻词）
```

### 多头注意力实现

```python
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_size=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # 为所有头一次性投影（更高效）
        self.qkv_projection = nn.Linear(embed_size, embed_size * 3)
        self.output_projection = nn.Linear(embed_size, embed_size)
        
        self.scale = self.head_dim ** -0.5
        
    def split_heads(self, x, batch_size):
        """
        将最后的维度拆分为多头
        
        输入：(batch, seq_len, embed_size)
        输出：(batch, num_heads, seq_len, head_dim)
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 1. QKV 投影
        qkv = self.qkv_projection(query)  # (batch, seq_len, 3*embed_size)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # 2. 计算注意力
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 3. 加权求和
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, value)
        
        # 4. 合并多头
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        
        # 5. 输出投影
        output = self.output_projection(context)
        
        return output, attention_weights

# 测试
mha = MultiHeadAttention(embed_size=512, num_heads=8)
Q = K = V = torch.randn(4, 100, 512)
output, attn_weights = mha(Q, K, V)
print(f"输出形状：{output.shape}")  # (4, 100, 512)
print(f"注意力权重形状：{attn_weights.shape}")  # (4, 8, 100, 100)
```

---

## 4. 注意力可视化

### 注意力权重可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, head_idx=0, ax=None):
    """
    可视化注意力权重热力图
    
    参数：
    - attention_weights: (batch, heads, seq_len, seq_len)
    - tokens: 词列表
    - head_idx: 要可视化的头索引
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 取第一个样本、指定头的注意力
    attn = attention_weights[0, head_idx].cpu().detach().numpy()
    
    # 绘制热力图
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 添加数值
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attn[i, j]:.2f}',
                          ha='center', va='center', color='gray', fontsize=8)
    
    # 颜色条
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    ax.set_title(f'Attention Head {head_idx}')
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    
    return ax

# 使用示例
tokens = ['我', '爱', '中', '国']
# 假设有注意力权重
attn_weights = torch.randn(1, 8, 4, 4)  # (batch, heads, seq_len, seq_len)
attn_weights = torch.softmax(attn_weights, dim=-1)  # 归一化

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    if i < 8:  # 8 个头
        visualize_attention(attn_weights, tokens, head_idx=i, ax=ax)
plt.tight_layout()
plt.show()
```

### 注意力模式分析

```python
def analyze_attention_patterns(attention_weights, tokens):
    """分析注意力模式"""
    
    # 平均注意力（所有头的平均）
    avg_attention = attention_weights[0].mean(dim=0).cpu().detach().numpy()
    
    print("=== 注意力分析 ===\n")
    
    for i, token in enumerate(tokens):
        # 该位置对其他位置的平均注意力
        attn_from_token = avg_attention[i]
        
        # 找出最关注的位置
        top_k_indices = attn_from_token.argsort()[::-1][:3]
        
        print(f"'{token}' 最关注:")
        for j in top_k_indices:
            if i != j:  # 排除自身
                print(f"  - '{tokens[j]}': {attn_from_token[j]:.3f}")
        print()

# 使用示例
tokens = ['我', '爱', '机', '器', '学', '习']
attn_weights = torch.randn(1, 8, len(tokens), len(tokens))
attn_weights = torch.softmax(attn_weights, dim=-1)

analyze_attention_patterns(attn_weights, tokens)
```

---

## 💻 实践项目

### 项目 1：实现完整自注意力模块

```python
"""
项目：完整的自注意力模块（含多头、掩码、残差）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（完整实现）"""
    
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert embed_size % num_heads == 0
        
        # QKV 投影
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_size, embed_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        参数：
        - x: (batch, seq_len, embed_size)
        - mask: 可选的注意力掩码
        
        返回：
        - output: (batch, seq_len, embed_size)
        - attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 合并头
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_size)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output, attention_weights


# 测试
embed_size = 512
num_heads = 8
batch_size = 4
seq_len = 100

model = MultiHeadSelfAttention(embed_size, num_heads)
x = torch.randn(batch_size, seq_len, embed_size)
output, attn = model(x)

print(f"输入形状：{x.shape}")
print(f"输出形状：{output.shape}")
print(f"注意力形状：{attn.shape}")
```

### 项目 2：注意力机制对比实验

```python
"""
项目：对比不同注意力变体的效果
"""

def compare_attention_variants():
    """比较不同注意力变体"""
    
    # 1. 标准自注意力
    class StandardAttention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.out = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
            attn = attn.softmax(dim=-1)
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.out(x)
    
    # 2. 线性注意力（近似）
    class LinearAttention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.out = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # 线性注意力：使用核函数近似 softmax
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            
            # (Q @ K^T) @ V = Q @ (K^T @ V)
            kv = k.transpose(-2, -1) @ v
            x = (q @ kv).transpose(1, 2).reshape(B, N, C)
            return self.out(x)
    
    # 性能对比
    import time
    
    d_model = 512
    seq_len = 1000
    x = torch.randn(4, seq_len, d_model)
    
    models = {
        'Standard': StandardAttention(d_model),
        'Linear': LinearAttention(d_model)
    }
    
    for name, model in models.items():
        # 预热
        for _ in range(10):
            model(x)
        
        # 计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"{name} Attention: {elapsed*1000:.2f}ms (100 次迭代)")

compare_attention_variants()
```

---

## ✅ 检查清单

- [ ] 理解注意力机制的核心公式
- [ ] 掌握 Q-K-V 的含义和计算
- [ ] 能够实现自注意力机制
- [ ] 理解多头注意力的优势
- [ ] 能够可视化注意力权重
- [ ] 完成实践项目

---

> _注意力是 Transformer 的灵魂，它让模型学会了"聚焦"。_
> 
> _—— 悟空_
