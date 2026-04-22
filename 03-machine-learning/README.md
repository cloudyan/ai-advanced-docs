# 第 3 章：机器学习基础（4-6 周）

> 从数据中学习规律 —— 经典机器学习算法全景
> 
> _学习周期：4-6 周 | 难度：⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 机器学习 vs 深度学习

```
机器学习：
  特征工程 + 经典算法（决策树、SVM、随机森林...）
  优点：可解释性强、小数据表现好
  缺点：需要人工特征工程

深度学习：
  端到端学习（神经网络自动学习特征）
  优点：无需人工特征、大数据表现极佳
  缺点：需要大量数据、黑盒模型

学习建议：先学机器学习，再学深度学习
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 理解监督学习和无监督学习的区别
- ✅ 掌握经典机器学习算法的原理
- ✅ 使用 scikit-learn 完成完整 ML 流程
- ✅ 进行模型评估和调优
- ✅ 独立完成 Kaggle 入门竞赛

---

## 📚 学习大纲

### 3.1 监督学习（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 什么是监督学习？

```
定义：从有标签的数据中学习映射关系

输入：X = [特征 1, 特征 2, ..., 特征 n]
输出：y = 标签（类别或数值）

任务类型：
- 分类：预测离散类别（如垃圾邮件/非垃圾邮件）
- 回归：预测连续数值（如房价、温度）
```

#### 核心算法详解

**1. 线性回归**

```python
# 原理：拟合一条直线 y = wx + b
# 目标：最小化预测值与真实值的平方误差

from sklearn.linear_model import LinearRegression
import numpy as np

# 数据
X = np.array([[1], [2], [3], [4], [5]])  # 特征
y = np.array([2, 4, 5, 4, 5])             # 目标

# 训练
model = LinearRegression()
model.fit(X, y)

# 预测
print(f"权重：{model.coef_[0]:.4f}")
print(f"偏置：{model.intercept_:.4f}")
print(f"预测 X=6: {model.predict([[6]])[0]:.4f}")
```

**2. 逻辑回归（分类）**

```python
# 原理：用 Sigmoid 函数将线性输出映射到 (0,1)
# P(y=1|x) = σ(wx + b) = 1 / (1 + e^(-(wx+b)))

from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])  # 二分类标签

model = LogisticRegression()
model.fit(X, y)

# 预测概率
proba = model.predict_proba([[3.5]])
print(f"属于类别 1 的概率：{proba[0][1]:.4f}")

# 预测类别
pred = model.predict([[3.5]])
print(f"预测类别：{pred[0]}")
```

**3. 决策树**

```python
# 原理：通过一系列 if-else 规则进行分类
# 分裂标准：信息增益或基尼系数

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X = np.array([
    [25, 50000],   # 年龄，收入
    [35, 80000],
    [45, 120000],
    [22, 30000],
    [50, 150000]
])
y = np.array([0, 1, 1, 0, 1])  # 是否购买

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

# 可视化
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['年龄', '收入'], 
          class_names=['不购买', '购买'], filled=True)
plt.show()

# 特征重要性
print(f"特征重要性：{model.feature_importances_}")
```

**4. 随机森林**

```python
# 原理：多棵决策树投票（集成学习）
# 优点：减少过拟合，提高泛化能力

from sklearn.ensemble import RandomForestClassifier

X = np.random.randn(1000, 20)  # 1000 样本，20 特征
y = np.random.randint(0, 2, 1000)

model = RandomForestClassifier(
    n_estimators=100,    # 100 棵树
    max_depth=10,        # 最大深度
    random_state=42
)
model.fit(X, y)

print(f"训练集准确率：{model.score(X, y):.4f}")
```

**5. 支持向量机（SVM）**

```python
# 原理：找到最大间隔的超平面
# 优点：高维空间表现好，有理论保证

from sklearn.svm import SVC

X = np.array([
    [1, 2], [2, 3], [3, 3],  # 类别 0
    [6, 5], [7, 7], [8, 6]   # 类别 1
])
y = np.array([0, 0, 0, 1, 1, 1])

model = SVC(kernel='rbf', C=1.0)
model.fit(X, y)

# 支持向量
print(f"支持向量数量：{len(model.support_vectors_)}")
```

---

#### 算法对比表

| 算法 | 适用场景 | 优点 | 缺点 | 复杂度 |
|------|---------|------|------|--------|
| 线性回归 | 回归任务 | 简单、可解释 | 只能拟合线性关系 | O(n×d) |
| 逻辑回归 | 二分类 | 简单、输出概率 | 线性边界 | O(n×d) |
| 决策树 | 分类/回归 | 可解释、无需归一化 | 容易过拟合 | O(n×d×logn) |
| 随机森林 | 分类/回归 | 准确率高、不易过拟合 | 模型大、速度慢 | O(k×n×d×logn) |
| SVM | 小样本分类 | 高维有效、理论保证 | 大样本慢、参数敏感 | O(n²×d) |
| KNN | 多分类 | 简单、无需训练 | 预测慢、对异常值敏感 | O(n×d) |

</details>

---

### 3.2 无监督学习（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 什么是无监督学习？

```
定义：从无标签的数据中发现结构

输入：X = [特征 1, 特征 2, ..., 特征 n]
输出：数据结构（聚类、降维结果）

任务类型：
- 聚类：将相似样本分组
- 降维：减少特征数量
- 异常检测：找出异常样本
```

#### K-Means 聚类

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成数据
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# 训练
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.title('K-Means 聚类结果')
plt.show()

# 评估（轮廓系数）
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
print(f"轮廓系数：{score:.4f}")  # 越接近 1 越好
```

#### PCA 降维

```python
from sklearn.decomposition import PCA

# 高维数据
X = np.random.randn(1000, 50)  # 1000 样本，50 维

# 降维到 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"原始维度：{X.shape}")
print(f"降维后维度：{X_pca.shape}")
print(f"解释方差比：{pca.explained_variance_ratio_}")
print(f"累计解释方差：{sum(pca.explained_variance_ratio_):.4f}")

# 可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA 降维结果')
plt.show()
```

#### t-SNE 可视化

```python
from sklearn.manifold import TSNE

# t-SNE 适合高维数据可视化
X = np.random.randn(1000, 50)
labels = np.random.randint(0, 4, 1000)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('t-SNE 可视化')
plt.show()
```

</details>

---

### 3.3 模型评估与调优（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 数据集划分

```python
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% 测试集
    random_state=42,
    stratify=y          # 分层抽样
)

print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")
```

#### 分类指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

y_true = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
y_pred = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]

print(f"准确率：{accuracy_score(y_true, y_pred):.4f}")
print(f"精确率：{precision_score(y_true, y_pred):.4f}")
print(f"召回率：{recall_score(y_true, y_pred):.4f}")
print(f"F1 分数：{f1_score(y_true, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print(f"混淆矩阵:\n{cm}")
```

#### 交叉验证

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5 折交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"各折准确率：{scores}")
print(f"平均准确率：{scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

#### 网格搜索调参

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
print(f"最佳分数：{grid_search.best_score_:.4f}")
```

</details>

---

### 3.4 特征工程（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 特征缩放

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化（均值为 0，方差为 1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化（缩放到 [0, 1]）
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
```

#### 类别特征编码

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Label Encoding（有序类别）
le = LabelEncoder()
colors = ['红', '蓝', '绿', '红', '蓝']
encoded = le.fit_transform(colors)
print(f"Label 编码：{encoded}")  # [2, 0, 1, 2, 0]

# One-Hot Encoding（无序类别）
ohe = OneHotEncoder()
categories = np.array([['红'], ['蓝'], ['绿'], ['红']])
one_hot = ohe.fit_transform(categories)
print(f"One-Hot 编码:\n{one_hot.toarray()}")
```

#### 处理缺失值

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([
    [1, 2, np.nan],
    [4, np.nan, 6],
    [7, 8, 9]
])

# 用均值填充
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

# 其他策略：'median'（中位数）、'most_frequent'（众数）、'constant'（常数）
```

#### 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择最重要的 k 个特征
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

print(f"选择的特征掩码：{selector.get_support()}")
print(f"原始特征数：{X.shape[1]}")
print(f"选择的特征数：{X_new.shape[1]}")
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 3.1 监督学习 | 2 周 | - | 0% | ⏳ |
| 3.2 无监督学习 | 1 周 | - | 0% | ⏳ |
| 3.3 模型评估 | 1 周 | - | 0% | ⏳ |
| 3.4 特征工程 | 1 周 | - | 0% | ⏳ |

### 项目清单

- [ ] 线性回归预测房价
- [ ] 决策树分类
- [ ] 随机森林 Kaggle 竞赛
- [ ] K-Means 客户分群
- [ ] PCA 降维可视化
- [ ] 完整 ML 流程（含调参）

---

## 📖 子章节索引

| 编号 | 章节 | 内容 | 状态 |
|------|------|------|------|
| 3.1 | 监督学习 | 线性回归、逻辑回归、决策树、随机森林、SVM | 📝 |
| 3.2 | 无监督学习 | K-Means、PCA、t-SNE | 📝 |
| 3.3 | 模型评估 | 指标、交叉验证、网格搜索 | 📝 |
| 3.4 | 特征工程 | 缩放、编码、缺失值处理 | 📝 |

---

> _机器学习是从数据中提取智慧的艺术，好的特征胜过复杂的模型。_
> 
> _—— 悟空_
