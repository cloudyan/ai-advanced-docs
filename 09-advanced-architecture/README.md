# 第 9 章：前沿架构与技术（8-10 周）

> 追踪 AI 最前沿 —— 从高效注意力到多模态大模型
> 
> _学习周期：8-10 周 | 难度：⭐⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐_

---

## 📖 本章概述

### 前沿技术全景图

```
2023-2024 AI 前沿技术：

┌─────────────────────────────────────────────────────────────────┐
│                    前沿技术分类                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. 高效注意力                                                   │
│    • FlashAttention: IO 感知注意力，2-4 倍加速                    │
│    • Sparse Attention: 稀疏注意力，O(n)复杂度                   │
│    • Linear Attention: 线性注意力                               │
│                                                                 │
│ 2. 长上下文处理                                                 │
│    • RoPE: 旋转位置编码，支持外推                               │
│    • Context Extension: 上下文扩展技术                          │
│    • Window Attention: 窗口注意力                               │
│                                                                 │
│ 3. 多模态大模型                                                 │
│    • CLIP: 图文对比学习                                         │
│    • LLaVA: 视觉语言助手                                        │
│    • GPT-4V: 多模态理解                                         │
│                                                                 │
│ 4. 推理加速架构                                                 │
│    • Speculative Decoding: 推测解码                             │
│    • Medusa: 多 token 预测                                      │
│    • Lookahead: 前瞻解码                                       │
│                                                                 │
│ 5. 高效架构                                                     │
│    • MoE: 混合专家模型                                          │
│    • RWKV: RNN+Transformer                                      │
│    • Mamba: 状态空间模型                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 理解 FlashAttention 的 IO 优化原理
- ✅ 实现 RoPE 位置编码
- ✅ 构建多模态 VQA 系统
- ✅ 使用推测解码加速推理
- ✅ 追踪并复现前沿论文

---

## 📚 学习大纲

### 9.1 高效注意力（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### FlashAttention 原理

```
传统注意力的问题：
1. 需要存储 N×N 的注意力矩阵 → O(N²) 显存
2. 多次 HBM 访问（高带宽内存）→ IO 瓶颈

FlashAttention 的优化：
1. 分块计算（Tiling）：不存储完整注意力矩阵
2. 重计算（Recomputation）：用计算换显存
3. IO 感知：最小化 HBM 访问次数

效果：
- 显存：从 O(N²) 降到 O(N)
- 速度：提升 2-4 倍
- 支持更长序列
```

#### FlashAttention 使用

```python
# 使用 FlashAttention 2
from flash_attn import flash_attn_func

# 输入：Q, K, V
# Q, K, V: (batch, seq_len, num_heads, head_dim)
q = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)

# FlashAttention
output = flash_attn_func(q, k, v, dropout_p=0.0)

# 对比标准注意力
def standard_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# 性能对比
import time

# FlashAttention
start = time.time()
for _ in range(100):
    output_flash = flash_attn_func(q, k, v, dropout_p=0.0)
torch.cuda.synchronize()
print(f"FlashAttention: {time.time() - start:.4f}s")

# Standard Attention
start = time.time()
for _ in range(100):
    output_std = standard_attention(q, k, v)
torch.cuda.synchronize()
print(f"Standard Attention: {time.time() - start:.4f}s")

# 显存对比
print(f"FlashAttention 显存：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

#### 稀疏注意力（Sparse Attention）

```python
"""
稀疏注意力模式：

1. 固定模式稀疏：
   ┌─────────────┐
   │█░░░░░░░░░░░░│  ← 局部窗口
   │░█░░░░░░░░░░░│
   │░░█░░░░░░░░░░│
   │░░░█░░░░░░░░░│
   └─────────────┘

2.  strides 稀疏：
   ┌─────────────┐
   │█░█░█░█░█░█░█│  ← 每隔 k 个关注
   │░█░█░█░█░█░█░│
   │█░█░█░█░█░█░█│
   └─────────────┘

3. 随机稀疏：
   ┌─────────────┐
   │█░░█░░░█░░░░░│  ← 随机选择
   │░░█░░░█░░░█░░│
   │░█░░░█░░░█░░░│
   └─────────────┘

优势：
- 复杂度从 O(N²) 降到 O(N) 或 O(N log N)
- 支持超长序列（100K+）
"""

# 使用 Longformer 的稀疏注意力
from transformers import LongformerModel

model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
# 支持 4096+ 的序列长度
```

</details>

---

### 9.2 长上下文处理（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### RoPE 旋转位置编码

```python
"""
RoPE（Rotary Positional Embedding）原理：

传统位置编码：
- 绝对位置编码：学习或固定的位置向量
- 问题：外推能力差（训练 512，推理 1024 就失效）

RoPE 的核心思想：
- 用旋转矩阵编码相对位置
- 保持注意力分数的相对位置信息
- 优秀的外推能力

数学公式：
对于位置 m 和 n 的 token：
Q_m = R(m) · q_m  （旋转 m 角度）
K_n = R(n) · k_n  （旋转 n 角度）

注意力分数：Q_m · K_n = q_m^T · R(n-m) · k_n
只依赖于相对位置 (n-m)！
"""

import torch

def apply_rope(q, k, freqs_cis):
    """
    应用 RoPE
    
    参数：
    - q, k: (batch, seq_len, num_heads, head_dim)
    - freqs_cis: 预计算的频率复数 (seq_len, head_dim/2)
    """
    # 转换为复数
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    # 旋转（复数乘法）
    q_rotated = q_complex * freqs_cis[:q.shape[1]].unsqueeze(1)
    k_rotated = k_complex * freqs_cis[:k.shape[1]].unsqueeze(1)
    
    # 转回实数
    q_out = torch.view_as_real(q_rotated).flatten(3)
    k_out = torch.view_as_real(k_rotated).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """预计算 RoPE 频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数
    return freqs_cis

# 测试
freqs_cis = precompute_freqs_cis(dim=128, max_len=2048)
q = torch.randn(2, 100, 8, 128)
k = torch.randn(2, 100, 8, 128)
q_rot, k_rot = apply_rope(q, k, freqs_cis)
print(f"RoPE 后形状：q={q_rot.shape}, k={k_rot.shape}")
```

#### 上下文扩展技术

```python
"""
上下文扩展方法：

1. 线性插值（Linear Interpolation）
   - 训练时位置：0, 1, 2, ..., 511
   - 推理时位置：0, 0.5, 1, 1.5, ..., 1023
   - 简单有效，但有精度损失

2. NTK 感知插值
   - 基于 NTK（Neural Tangent Kernel）理论
   - 动态调整不同频率的插值方式
   - 更好的外推效果

3. YaRN（Yet another RoPE scaling）
   - 结合插值和微调
   - 支持 128K+ 上下文
"""

# NTK 感知插值实现
def ntK_interpolation(freqs_cis, scale_factor, original_max_len=4096):
    """
    NTK 感知位置编码插值
    
    参数：
    - freqs_cis: 原始频率
    - scale_factor: 扩展倍数（如 2 表示扩展到 2 倍）
    - original_max_len: 原始最大长度
    """
    # 获取原始频率
    freqs = torch.angle(freqs_cis)
    
    # NTK 调整
    for i in range(len(freqs)):
        if freqs[i] > original_max_len / (2 * torch.pi):
            freqs[i] = freqs[i] / scale_factor
    
    # 重新构建复数
    freqs_cis_scaled = torch.exp(1j * freqs)
    return freqs_cis_scaled

# 使用示例
original_freqs = precompute_freqs_cis(128, 4096)
extended_freqs = ntK_interpolation(original_freqs, scale_factor=4)
# 现在支持 4096 * 4 = 16384 的上下文
```

</details>

---

### 9.3 多模态大模型（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### CLIP 原理

```
CLIP（Contrastive Language-Image Pre-training）：

架构：
┌─────────────┐    ┌─────────────┐
│  图像编码器  │    │  文本编码器  │
│  (ViT/RN)   │    │ (Transformer)│
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
   图像特征 (128 维)    文本特征 (128 维)
       │                  │
       └────────┬─────────┘
                │
          对比学习损失
          
训练目标：
- 匹配的图文对：特征相似度高
- 不匹配的图文对：特征相似度低

应用：
- 零样本图像分类
- 图文检索
- 多模态理解基础
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像编码
image = Image.open("cat.jpg")
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)

# 文本编码
text_inputs = processor(
    text=["一只猫", "一只狗", "一辆车"],
    return_tensors="pt",
    padding=True
)
text_features = model.get_text_features(**text_inputs)

# 计算相似度
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).softmax(dim=-1)
print(f"相似度：{similarity}")
# 输出：[[0.85, 0.10, 0.05]] - 最可能是猫
```

#### LLaVA 多模态对话

```python
"""
LLaVA（Large Language and Vision Assistant）：

架构：
┌─────────────┐    ┌─────────────┐
│  视觉编码器  │    │  语言模型   │
│  (CLIP ViT) │───▶│  (LLaMA)    │
│             │    │             │
└─────────────┘    └─────────────┘
       │                   │
       └───────────────────┘
              投影层

特点：
- 端到端多模态对话
- 视觉问答（VQA）
- 图像理解 + 语言生成
"""

# 使用 LLaVA
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.conversation import conv_templates

# 加载模型
model_path = "llava-hf/llava-1.5-7b-hf"
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path)

# 准备图像和对话
image = Image.open("image.jpg")
conv = conv_templates["vicuna_v1"].copy()

# 第一轮：用户提问
conv.append_message(conv.roles[0], "这张图片里有什么？")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# 处理输入
inputs = tokenizer([prompt], return_tensors="pt")
image_tensor = process_images([image], image_processor, model.config)

# 生成回答
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        images=image_tensor,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"LLaVA 回答：{response}")
```

</details>

---

### 9.4 推理加速架构（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 推测解码（Speculative Decoding）

```
推测解码原理：

传统解码：
Token1 → Token2 → Token3 → Token4 → ...
  ↓       ↓       ↓       ↓
串行生成，慢

推测解码：
小模型（草稿）: Token1 → Token2 → Token3 → Token4
                  ↓
大模型（验证）:   一次性验证所有 token
                  ↓
              接受或拒绝

优势：
- 如果小模型预测准确，可以一次生成多个 token
- 加速比：2-4 倍（取决于接受率）
"""

# 使用 Transformers 的推测解码
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载大小模型
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

input_text = "人工智能的未来"
inputs = tokenizer(input_text, return_tensors="pt")

# 推测解码生成
with torch.no_grad():
    outputs = target_model.generate(
        **inputs,
        assistant_model=draft_model,  # 指定草稿模型
        num_assistant_tokens=5,       # 每次生成 5 个草稿 token
        max_new_tokens=50
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### Medusa 多 token 预测

```python
"""
Medusa 原理：

传统解码：
每次生成 1 个 token → 需要 N 次前向传播

Medusa：
- 添加多个解码头
- 每个头预测不同位置的 token
- 一次前向传播生成多个 token

架构：
        ┌─────────────┐
        │  LLM Backbone│
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
 Head 0     Head 1     Head 2
(t+1)      (t+2)      (t+3)

加速比：2-3 倍
"""

# Medusa 使用示例（伪代码）
from medusa.model import MedusaModel

model = MedusaModel.from_pretrained("FasterDecoding/medusa-7b")

# Medusa 有多个解码头
# head[0]: 预测下一个 token
# head[1]: 预测下下个 token
# head[2]: 预测下下下个 token

outputs = model.generate(
    input_ids,
    medusa_num_heads=3,      # 3 个解码头
    max_new_tokens=100
)
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 9.1 高效注意力 | 2 周 | - | 0% | ⏳ |
| 9.2 长上下文 | 2 周 | - | 0% | ⏳ |
| 9.3 多模态 | 2 周 | - | 0% | ⏳ |
| 9.4 推理加速 | 2 周 | - | 0% | ⏳ |

### 项目清单

- [ ] FlashAttention 性能对比
- [ ] RoPE 位置编码实现
- [ ] CLIP 零样本分类
- [ ] LLaVA 多模态对话
- [ ] 推测解码加速实验

---

> _前沿是明天的基础，今天读懂论文，明天创造论文。_
> 
> _—— 悟空_
