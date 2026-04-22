# 🚀 部署指南

## 步骤 1：创建 GitHub 仓库

1. 访问 https://github.com/new
2. 仓库名：`ai-advanced-docs`
3. 可见性：**Public**
4. 点击 **Create repository**

---

## 步骤 2：推送代码

在终端执行：

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-advanced-docs

# 如果已经初始化过
git remote add origin git@github.com:cloudyan/ai-advanced-docs.git
git push -u origin main

# 或重新推送
git push -f -u origin main
```

---

## 步骤 3：启用 GitHub Pages

1. 访问：https://github.com/cloudyan/ai-advanced-docs/settings/pages
2. **Source**: Deploy from a branch
3. **Branch**: main / (root)
4. 点击 **Save**

等待 1-2 分钟即可访问！

---

## 🎉 完成后的访问地址

```
https://cloudyan.github.io/ai-advanced-docs/
```

---

## 📖 文档结构

```
ai-advanced-docs/
├── index.html              # Docsify 主配置
├── README.md               # 首页
├── _sidebar.md             # 侧边栏导航
├── deploy.sh               # 部署脚本
├── 01-math-foundation/     # 数学基础
├── 02-python-basics/       # Python 基础
├── 03-machine-learning/    # 机器学习
├── 04-deep-learning/       # 深度学习
├── 05-transformer/         # Transformer ⭐
├── 06-llm-application/     # LLM 应用
├── 07-ai-agent/            # AI Agent
├── 08-system-optimization/ # 系统优化
├── 09-advanced-architecture/ # 高级架构
├── 10-research/            # 研究
├── 11-engineering/         # 工程
└── 12-innovation/          # 创新
```

---

*最后更新：2026-04-22*
