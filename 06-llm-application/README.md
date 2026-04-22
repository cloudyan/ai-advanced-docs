# 第 6 章：大模型应用开发（6-8 周）

> 站在巨人肩膀上创新 —— Prompt、RAG、微调实战
> 
> _学习周期：6-8 周 | 难度：⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 大模型应用开发三驾马车

```
┌─────────────────────────────────────────────────────────────────┐
│                    大模型应用开发                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Prompt    │    │     RAG     │    │   微调      │         │
│  │   工程      │    │  检索增强   │    │  Fine-tuning│         │
│  │             │    │  生成       │    │             │         │
│  │ • 零样本    │    │ • 向量检索  │    │ • 全量微调  │         │
│  │ • 少样本    │    │ • 知识库    │    │ • LoRA      │         │
│  │ • 思维链    │    │ • 上下文增强│    │ • QLoRA     │         │
│  │             │    │             │    │             │         │
│  │ 成本：$     │    │ 成本：$$    │    │ 成本：$$$   │         │
│  │ 效果：★★★   │    │ 效果：★★★★  │    │ 效果：★★★★★ │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  选择建议：                                                      │
│  1. 先用 Prompt 解决（最快、最便宜）                              │
│  2. 不行再加 RAG（需要外部知识）                                 │
│  3. 最后考虑微调（领域专用、高质量数据）                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 设计高效的 Prompt 解决复杂任务
- ✅ 构建 RAG 系统实现知识库问答
- ✅ 使用 LoRA 微调大模型
- ✅ 使用 LangChain 开发 AI 应用
- ✅ 独立完成企业级 AI 应用开发

---

## 📚 学习大纲

### 6.1 Prompt 工程（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Prompt 基础

```python
# 好的 Prompt 三要素
good_prompt = """
# 角色设定
你是一位资深的数据科学家，擅长机器学习教学。

# 任务描述
请解释什么是过拟合，要求：
1. 用通俗易懂的类比
2. 给出一个具体例子
3. 说明如何避免

# 输出格式
用 Markdown 格式，包含代码示例
"""

# 对比：差的 Prompt
bad_prompt = "什么是过拟合？"
```

#### Zero-shot vs Few-shot

```python
# Zero-shot（无示例）
zero_shot = """
将以下英文翻译成中文：
"The quick brown fox jumps over the lazy dog."
"""

# Few-shot（少样本示例）
few_shot = """
翻译示例：
English: Hello, how are you?
Chinese: 你好，最近怎么样？

English: I love machine learning.
Chinese: 我喜欢机器学习。

English: The weather is nice today.
Chinese: 
"""
# 模型会自动续写：今天天气很好。
```

#### 思维链（Chain of Thought）

```python
# 普通 Prompt
cot_normal = """
小明有 5 个苹果，他给了小红 2 个，又买了 3 个，现在有几个？
"""

# CoT Prompt（引导逐步思考）
cot_prompt = """
小明有 5 个苹果，他给了小红 2 个，又买了 3 个，现在有几个？

请逐步思考：
1. 初始有多少苹果？
2. 给了小红后剩多少？
3. 买了 3 个后是多少？
4. 最终答案是多少？
"""

# 效果对比
# 普通 Prompt 可能直接回答，容易出错
# CoT Prompt 引导模型展示推理过程，准确率更高
```

#### ReAct 模式（Reason + Act）

```python
react_prompt = """
你是一个智能助手，可以使用工具解决问题。

可用工具：
- search(query): 搜索网络
- calculate(expression): 计算数学表达式

问题：特斯拉 2023 年的营收是多少？同比增长多少？

思考过程：
Thought: 我需要先搜索特斯拉 2023 年的营收数据
Action: search("特斯拉 2023 年营收")
Observation: 特斯拉 2023 年营收 967.7 亿美元，同比增长 19%

Thought: 我已经获得了所需信息
Final Answer: 特斯拉 2023 年营收为 967.7 亿美元，同比增长 19%
"""
```

#### Prompt 优化技巧

```python
# 1. 结构化输出
structured = """
请分析以下文本的情感，按 JSON 格式输出：

{
  "sentiment": "positive/negative/neutral",
  "confidence": 0-1 之间的数字，
  "keywords": ["关键词 1", "关键词 2"]
}

文本：这个产品真的很好用，我非常喜欢！
"""

# 2. 角色设定
role_play = """
你是一位经验丰富的 Python 工程师，擅长代码审查。
请审查以下代码，指出：
1. 潜在 bug
2. 性能问题
3. 改进建议
"""

# 3. 分步指令
step_by_step = """
请按以下步骤完成任务：

步骤 1：阅读并理解输入文本
步骤 2：提取关键信息
步骤 3：总结核心观点
步骤 4：生成简洁摘要

输入：[长文本]
"""
```

</details>

---

### 6.2 RAG 检索增强生成（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### RAG 原理

```
RAG 工作流程：

用户问题 ──→ 向量化 ──→ 向量检索 ──→ 相关文档片段
                                              │
                                              ▼
用户问题 + 相关文档 ──→ LLM ──→ 增强生成答案
```

#### 向量数据库对比

| 数据库 | 特点 | 适用场景 | 难度 |
|--------|------|---------|------|
| FAISS | Facebook 开源，速度快 | 本地部署、小规模 | ⭐⭐ |
| Chroma | 轻量级，易上手 | 原型开发、小项目 | ⭐ |
| Milvus | 功能完整，可扩展 | 生产环境、大规模 | ⭐⭐⭐ |
| Pinecone | 托管服务 | 快速上线 | ⭐ |

#### 完整 RAG 实现

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = TextLoader("company_knowledge.txt", encoding='utf-8')
documents = loader.load()

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回最相关的 3 个文档
)

# 5. 创建 QA 链
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. 提问
query = "公司的年假政策是什么？"
result = qa_chain({"query": query})

print(f"答案：{result['result']}")
print(f"参考文档：{result['source_documents']}")
```

#### RAG 优化技巧

```python
# 1. 混合检索（稠密 + 稀疏）
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

# 稠密检索
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 稀疏检索（BM25）
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

# 混合
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# 2. 重排序（Rerank）
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# 3. 多查询检索
from langchain.retrievers.multi_query import MultiQueryRetriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

</details>

---

### 6.3 微调技术（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 微调方式对比

| 方式 | 可训练参数 | 显存需求 | 效果 | 适用场景 |
|------|-----------|---------|------|---------|
| 全量微调 | 100% | 极高 | 最佳 | 充足资源 |
| LoRA | 1-10% | 低 | 接近全量 | 推荐默认 |
| QLoRA | 1-10% | 极低 | 略低于 LoRA | 资源受限 |

#### LoRA 原理

```
LoRA 核心思想：
原始权重 W 冻结，训练低秩适配器

W' = W + ΔW
ΔW = A × B

其中：
- W: 原始权重（冻结）
- A: 下投影矩阵 (d×r)，可训练
- B: 上投影矩阵 (r×k)，可训练
- r: 秩（通常 4-64），远小于 d

优势：
- 可训练参数减少 10000 倍
- 效果接近全量微调
- 可快速切换不同任务
```

#### LoRA 微调实战（使用 PEFT）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

# 1. 加载模型和分词器
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 秩
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.1,
    target_modules=["query_key_value"],  # 目标模块
    bias="none",
    inference_mode=False
)

# 3. 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 4M || all params: 6000M || trainable%: 0.07%

# 4. 准备数据
train_data = [
    {"text": "问题：什么是机器学习？答案：机器学习是..."},
    {"text": "问题：Python 中如何实现列表推导？答案：..."},
    # ... 更多数据
]

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 6. 创建 Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

# 7. 开始训练
trainer.train()

# 8. 保存 LoRA 权重
model.save_pretrained("./lora_weights")
```

#### QLoRA 微调（量化 +LoRA）

```python
from transformers import BitsAndBytesConfig

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 然后同样应用 LoRA 配置
model = get_peft_model(model, lora_config)

# 训练方式相同
# 显存需求从 24GB+ 降低到 12GB 左右
```

#### 微调数据格式

```python
# 指令微调数据格式
train_data = [
    {
        "instruction": "请翻译以下句子",
        "input": "The quick brown fox jumps over the lazy dog.",
        "output": "敏捷的棕色狐狸跳过了懒惰的狗。"
    },
    {
        "instruction": "写一首关于春天的诗",
        "input": "",
        "output": "春风拂面花自开，\n柳绿桃红燕归来...\n"
    }
]

# 转换为训练格式
def format_example(example):
    if example.get("input"):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Output:
{example['output']}"""
```

</details>

---

### 6.4 大模型开发框架（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### LangChain 核心概念

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# 1. 模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 2. 消息
messages = [
    SystemMessage(content="你是一位有帮助的助手"),
    HumanMessage(content="你好，请介绍一下自己")
]

# 3. 调用
response = llm(messages)
print(response.content)
```

#### Chain（链）

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Prompt 模板
prompt = ChatPromptTemplate.from_template(
    "请用{style}的风格解释什么是{concept}"
)

# 链
chain = LLMChain(llm=llm, prompt=prompt)

# 使用
result = chain.invoke({"style": "幽默", "concept": "量子力学"})
print(result["text"])
```

#### Agent（智能体）

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper

# 工具
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当你需要搜索实时信息时使用"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行
result = agent.run("今天北京的天气如何？")
```

#### Memory（记忆）

```python
from langchain.memory import ConversationBufferMemory

# 带记忆的对话
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建带记忆的链
from langchain.chains import ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 多轮对话
print(conversation.predict(input="你好，我叫小明"))
print(conversation.predict(input="我今年 25 岁"))
print(conversation.predict(input="我叫什么名字？"))  # 能记住
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 6.1 Prompt 工程 | 2 周 | - | 0% | ⏳ |
| 6.2 RAG | 2 周 | - | 0% | ⏳ |
| 6.3 微调技术 | 2 周 | - | 0% | ⏳ |
| 6.4 开发框架 | 1 周 | - | 0% | ⏳ |

### 项目清单

- [ ] Prompt 模板库
- [ ] 企业知识库 RAG 系统
- [ ] LoRA 微调开源模型
- [ ] LangChain AI 助手

---

> _大模型是新的计算机，Prompt 是新的编程语言，RAG 是新的数据库，微调是新的编译器。_
> 
> _—— 悟空_
