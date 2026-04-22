# 第 2 章：Python 编程基础（2-3 周）

> 掌握 AI 开发的利器 —— 从 Python 基础到 PyTorch 实战
> 
> _学习周期：2-3 周 | 难度：⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 为什么选择 Python 做 AI？

```
┌─────────────────────────────────────────────────────────┐
│  Python 在 AI 领域的优势                                 │
├─────────────────────────────────────────────────────────┤
│  ✅ 简洁易学：语法清晰，上手快                           │
│  ✅ 生态丰富：NumPy、Pandas、PyTorch、TensorFlow        │
│  ✅ 社区活跃：问题容易找到解答                           │
│  ✅ 胶水语言：容易集成 C/C++ 高性能代码                    │
│  ✅ 数据科学友好：Jupyter、可视化库完善                  │
└─────────────────────────────────────────────────────────┘
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 熟练使用 Python 进行数据处理
- ✅ 掌握 NumPy 进行科学计算
- ✅ 使用 Pandas 进行数据分析
- ✅ 用 PyTorch 构建和训练神经网络
- ✅ 独立完成 MNIST 手写数字识别项目

---

## 📚 学习大纲

### 2.1 Python 核心（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Day 1-2: Python 基础语法

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| 变量与数据类型 | int、float、str、bool | 基础类型转换练习 |
| 运算符 | 算术、比较、逻辑、位运算 | 表达式计算 |
| 字符串操作 | 切片、格式化、常用方法 | 字符串处理练习 |
| 类型转换 | 显式/隐式转换 | 类型转换实践 |

**代码示例**：
```python
# 变量与数据类型
name = "悟空"           # str
age = 1000             # int
power_level = 9999.99  # float
is_immortal = True     # bool

# 字符串操作
greeting = f"我是{name}，今年{age}岁"
print(greeting.upper())  # 转大写

# 类型转换
num_str = "123"
num_int = int(num_str)  # str → int
```

---

#### Day 3-4: 控制流与数据结构

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| 条件语句 | if/elif/else | 条件判断练习 |
| 循环 | for/while、break/continue | 循环练习 |
| 列表 | 创建、索引、切片、方法 | 列表操作 |
| 字典 | 创建、访问、遍历 | 字典操作 |
| 集合与元组 | 去重、不可变序列 | 集合运算 |
| 推导式 | 列表/字典/集合推导式 | 推导式练习 |

**代码示例**：
```python
# 条件语句
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# 循环
for i in range(5):
    if i == 3:
        continue  # 跳过
    print(i)

# 列表操作
numbers = [1, 2, 3, 4, 5]
numbers.append(6)        # 添加
sliced = numbers[1:4]    # 切片 [2, 3, 4]
reversed = numbers[::-1] # 反转

# 字典
student = {"name": "悟空", "age": 1000}
print(student["name"])   # 访问
student["power"] = 9999  # 添加

# 推导式
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

---

#### Day 5-6: 函数与模块

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| 函数定义 | def、参数、返回值 | 函数编写 |
| 参数类型 | 位置参数、默认参数、可变参数 | 参数练习 |
| lambda 函数 | 匿名函数 | lambda 使用 |
| 模块导入 | import、from...import | 模块使用 |
| 包管理 | pip、requirements.txt | 包安装 |

**代码示例**：
```python
# 函数定义
def greet(name, greeting="你好"):
    """打招呼函数"""
    return f"{greeting}，{name}！"

print(greet("悟空"))           # 默认参数
print(greet("悟空", "Hello"))  # 指定参数

# 可变参数
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# lambda 函数
square = lambda x: x ** 2
print(square(5))  # 25

# 排序中使用 lambda
numbers = [1, 5, 2, 8, 3]
numbers.sort(key=lambda x: -x)  # 降序
```

---

#### Day 7: 面向对象与异常处理

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| 类与对象 | class、__init__、self | 类定义 |
| 继承与多态 | 父类、子类、重写 | 继承实践 |
| 魔术方法 | __str__、__repr__等 | 魔术方法 |
| 异常处理 | try/except/finally | 错误处理 |
| 文件操作 | 读写文本/JSON | 文件处理 |

**代码示例**：
```python
# 类定义
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}({self.age}岁)"
    
    def greet(self):
        return f"我是{self.name}"

# 继承
class Student(Person):
    def __init__(self, name, age, grade):
        super().__init__(name, age)
        self.grade = grade
    
    def greet(self):  # 重写
        return f"我是{self.name}，{self.grade}年级"

# 异常处理
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"错误：{e}")
finally:
    print("清理工作")

# 文件操作
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!")

with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

---

#### ✅ 实践项目：数据处理脚本

```python
"""
项目：CSV 数据清洗与统计
要求：
1. 读取 CSV 文件
2. 处理缺失值
3. 计算统计指标
4. 输出结果到新文件
"""

import csv
from datetime import datetime

def process_csv(input_file, output_file):
    """处理 CSV 数据"""
    data = []
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    # 处理缺失值、计算统计
    # ... (学员完成)
    
    # 输出结果
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    process_csv("input.csv", "output.csv")
```

</details>

---

### 2.2 科学计算栈（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Day 8-9: NumPy 数值计算

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| 数组创建 | array、zeros、ones、arange | 数组创建 |
| 数组操作 | 索引、切片、 reshape | 形状变换 |
| 广播机制 | 不同形状数组运算 | 广播练习 |
| 向量化 | 替代循环、性能优化 | 向量化实现 |
| 线性代数 | 点积、矩阵乘法、特征值 | 线性代数 |

**代码示例**：
```python
import numpy as np

# 数组创建
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))      # 3x3 零矩阵
ones = np.ones((2, 4))        # 2x4 全 1 矩阵
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 数组操作
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d.shape)      # (2, 3)
print(arr_2d.reshape(3, 2))  # 重塑
print(arr_2d[:, 1])      # 取第二列 [2, 5]

# 广播机制
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)  # 每行加 b

# 向量化（比循环快 100 倍+）
arr = np.arange(1000000)
# 推荐：向量化
result = arr ** 2
# 不推荐：循环
# result = [x**2 for x in arr]

# 线性代数
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))      # 矩阵乘法
print(np.linalg.inv(A))  # 逆矩阵
eigenvalues, eigenvectors = np.linalg.eig(A)  # 特征值分解
```

---

#### Day 10-11: Pandas 数据处理

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| DataFrame | 创建、索引、列操作 | DataFrame 操作 |
| 数据读取 | read_csv、read_excel | 文件读取 |
| 数据清洗 | 缺失值、重复值、异常值 | 数据清洗 |
| 数据筛选 | 条件筛选、query | 数据过滤 |
| 分组聚合 | groupby、agg | 分组统计 |
| 数据合并 | merge、concat | 数据拼接 |

**代码示例**：
```python
import pandas as pd

# DataFrame 创建
df = pd.DataFrame({
    'name': ['悟空', '八戒', '沙僧'],
    'age': [1000, 500, 800],
    'power': [9999, 8000, 7000]
})

# 数据读取
df = pd.read_csv('data.csv')

# 数据清洗
df.dropna()              # 删除缺失值
df.fillna(0)             # 填充缺失值
df.drop_duplicates()     # 去重

# 数据筛选
high_power = df[df['power'] > 8000]
result = df.query('age > 500 and power > 7500')

# 分组聚合
grouped = df.groupby('category')['power'].mean()
result = df.groupby('category').agg({
    'age': 'mean',
    'power': ['min', 'max', 'mean']
})

# 数据合并
merged = pd.merge(df1, df2, on='id')
concatenated = pd.concat([df1, df2])
```

---

#### Day 12-13: 数据可视化

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| Matplotlib 基础 | 折线图、散点图、柱状图 | 基础图表 |
| 子图与布局 | 多子图、figsize | 复杂布局 |
| Seaborn | 统计图表、热力图 | 高级图表 |
| 样式美化 | 颜色、标签、图例 | 图表美化 |

**代码示例**：
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 折线图
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('正弦函数')
plt.legend()
plt.grid(True)
plt.show()

# 多子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, np.sin(x))
axes[0, 1].scatter(x, np.cos(x))
axes[1, 0].bar(range(5), [1, 3, 2, 4, 5])
axes[1, 1].hist(np.random.randn(1000), bins=30)
plt.tight_layout()
plt.show()

# Seaborn 热力图
data = np.random.rand(10, 10)
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.show()

# 箱线图
sns.boxplot(data=df, x='category', y='value')
plt.show()
```

---

#### ✅ 实践项目：Kaggle Titanic 探索性分析

```python
"""
项目：泰坦尼克数据集探索性分析（EDA）
要求：
1. 加载数据并了解基本结构
2. 数据质量分析（缺失值、异常值）
3. 单变量分析（分布可视化）
4. 双变量分析（特征与生存率关系）
5. 总结发现
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('titanic.csv')

# 2. 数据质量分析
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. 单变量分析
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['Age'], ax=axes[0, 0])
sns.countplot(df['Sex'], ax=axes[0, 1])
sns.countplot(df['Pclass'], ax=axes[1, 0])
sns.histplot(df['Fare'], ax=axes[1, 1])
plt.tight_layout()
plt.show()

# 4. 双变量分析
# 性别与生存率
survival_by_sex = df.groupby('Sex')['Survived'].mean()
print(survival_by_sex)

# 舱位与生存率
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_by_class)

# 5. 总结发现
# 学员完成分析报告
```

</details>

---

### 2.3 PyTorch 深度学习框架（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Day 14-15: Tensor 基础

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| Tensor 创建 | 从列表、随机、特殊值 | Tensor 创建 |
| Tensor 操作 | 索引、切片、变形 | 形状操作 |
| 数学运算 | 加减乘除、矩阵运算 | 数学计算 |
| GPU 加速 | cuda、to 设备 | GPU 计算 |

**代码示例**：
```python
import torch

# Tensor 创建
t1 = torch.tensor([1, 2, 3, 4])           # 从列表
t2 = torch.zeros(3, 3)                     # 零矩阵
t3 = torch.ones(2, 4)                      # 全 1 矩阵
t4 = torch.rand(3, 3)                      # 随机 [0,1)
t5 = torch.randn(3, 3)                     # 标准正态分布
t6 = torch.arange(0, 10, 2)                # [0, 2, 4, 6, 8]

# Tensor 操作
print(t1.shape)        # torch.Size([4])
print(t1[1:3])         # 切片 [2, 3]
print(t1.reshape(2, 2)) # 变形
print(t1.unsqueeze(0))  # 增加维度 (1, 4)
print(t1.squeeze())     # 去除维度 1

# 数学运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)           # 加法 [5, 7, 9]
print(a * b)           # 逐元素乘法
print(torch.dot(a, b)) # 点积 32
print(torch.matmul(a, b))  # 矩阵乘法

# GPU 加速
if torch.cuda.is_available():
    device = torch.device('cuda')
    t_gpu = t1.to(device)
    print(t_gpu.device)  # cuda:0
```

---

#### Day 16-17: 自动求导与 nn.Module

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| autograd | 计算图、requires_grad | 自动求导 |
| 反向传播 | backward()、grad 属性 | 反向传播 |
| nn.Module | 模型基类、层定义 | 模型定义 |
| 常用层 | Linear、Conv2d、LSTM | 层使用 |

**代码示例**：
```python
import torch
import torch.nn as nn

# 自动求导
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7

# 定义神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.softmax(x)

# 实例化模型
model = NeuralNetwork(784, 128, 10)
print(model)

# 前向传播
input_tensor = torch.randn(32, 784)  # batch_size=32
output = model(input_tensor)
print(output.shape)  # torch.Size([32, 10])
```

---

#### Day 18-19: DataLoader 与训练流程

| 主题 | 知识点 | 练习任务 |
|------|--------|----------|
| Dataset | 自定义数据集 | 数据加载 |
| DataLoader | 批量加载、打乱、多进程 | 数据迭代 |
| 损失函数 | CrossEntropyLoss、MSELoss | 损失计算 |
| 优化器 | SGD、Adam | 参数更新 |
| 训练循环 | 完整训练流程 | 模型训练 |

**代码示例**：
```python
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 自定义 Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建 DataLoader
dataset = CustomDataset(train_data, train_labels)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 完整训练循环
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 数据移到设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()  # 清零梯度
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 训练多轮
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(10):
    loss, acc = train_epoch(model, dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.2f}%')
```

---

#### ✅ 实践项目：MNIST 手写数字识别

```python
"""
项目：MNIST 手写数字识别
目标：准确率 > 98%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    './data', train=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 2. 定义模型
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# 3. 训练
device = torch.device('cuda')
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 测试
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = 100. * correct / len(test_dataset)
    print(f'Epoch {epoch+1}: Accuracy={accuracy:.2f}%')

# 4. 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 天数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 2.1 Python 核心 | 7 天 | - | 0% | ⏳ |
| 2.2 科学计算栈 | 6 天 | - | 0% | ⏳ |
| 2.3 PyTorch 框架 | 6 天 | - | 0% | ⏳ |

### 自测清单

学完本章后，你应该能够：

- [ ] 熟练使用 Python 列表、字典、集合
- [ ] 编写函数和类
- [ ] 使用 NumPy 进行向量化计算
- [ ] 使用 Pandas 处理 CSV 数据
- [ ] 用 Matplotlib/Seaborn 可视化
- [ ] 创建 PyTorch Tensor 并进行运算
- [ ] 使用 autograd 自动求导
- [ ] 定义 nn.Module 模型
- [ ] 编写完整训练循环
- [ ] MNIST 准确率达到 98%+

---

## 📖 子章节索引

| 编号 | 章节 | 内容 | 状态 |
|------|------|------|------|
| 2.1 | Python 核心 | Day 1-7 详细内容 | 📝 |
| 2.2 | 科学计算栈 | Day 8-13 详细内容 | 📝 |
| 2.3 | PyTorch 框架 | Day 14-19 详细内容 | 📝 |

---

> _代码是 AI 工程师的画笔，熟练运用才能创作出精彩的作品。_
> 
> _—— 悟空_
