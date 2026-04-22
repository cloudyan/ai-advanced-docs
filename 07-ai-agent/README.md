# 第 7 章：AI Agent（6-8 周）

> 构建自主智能体 —— 从感知到行动，从单兵到协作
> 
> _学习周期：6-8 周 | 难度：⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 什么是 AI Agent？

```
传统 AI：输入 → 模型 → 输出（一次性）

AI Agent：感知 → 思考 → 行动 → 观察 → 循环...
              ↑                    │
              └────────────────────┘
                   持续学习与改进
```

### Agent 的核心能力

| 能力 | 描述 | 示例 |
|------|------|------|
| **感知** | 获取环境信息 | 读取文件、搜索网络、API 调用 |
| **思考** | 分析、规划、推理 | 任务分解、策略制定 |
| **行动** | 执行操作改变环境 | 写代码、调用工具、发送消息 |
| **记忆** | 存储和检索信息 | 对话历史、知识库、经验 |
| **学习** | 从经验中改进 | 反思、技能积累 |

### 本章学习目标

学完本章后，你将能够：
- ✅ 理解 Agent 的架构设计原理
- ✅ 实现具备规划能力的 Agent
- ✅ 为 Agent 集成多种工具
- ✅ 构建自进化 Agent（能从失败中学习）
- ✅ 设计多 Agent 协作系统

---

## 📚 学习大纲

### 7.1 Agent 基础（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Agent 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Agent 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │   感知层    │      │   思考层    │      │   行动层    │     │
│  │             │      │             │      │             │     │
│  │ • 用户输入  │─────▶│ • 任务理解  │─────▶│ • 工具调用  │     │
│  │ • 环境观察  │      │ • 规划分解  │      │ • 执行操作  │     │
│  │ • 工具返回  │      │ • 推理决策  │      │ • 输出结果  │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│         ▲                    │                    │            │
│         │                    ▼                    │            │
│         │            ┌─────────────┐              │            │
│         │            │   记忆层    │              │            │
│         │            │             │◀─────────────┘            │
│         │            │ • 短期记忆  │                           │
│         └────────────│ • 长期记忆  │                           │
│                      │ • 技能库    │                           │
│                      └─────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 基础 Agent 实现

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    SEARCH = "search"
    CODE = "code"
    API_CALL = "api_call"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    MESSAGE = "message"

@dataclass
class Action:
    type: ActionType
    params: Dict
    description: str

@dataclass
class Observation:
    success: bool
    result: str
    error: Optional[str] = None

class BaseAgent:
    """基础 Agent 类"""
    
    def __init__(self, llm, tools: Dict = None):
        self.llm = llm
        self.tools = tools or {}
        self.memory = []  # 对话历史
        self.max_iterations = 10
        
    def run(self, task: str) -> str:
        """执行任务的主循环"""
        self.memory.append({"role": "user", "content": task})
        
        for iteration in range(self.max_iterations):
            # 1. 思考：决定下一步行动
            thought = self._think()
            
            # 2. 行动：执行决定的行动
            if thought.action:
                observation = self._act(thought.action)
                self.memory.append({
                    "role": "observation",
                    "content": observation.result
                })
                
                # 3. 判断是否完成
                if thought.is_complete:
                    return thought.final_answer
            else:
                # 没有行动，直接回答
                return thought.response
        
        return "达到最大迭代次数，任务未完成"
    
    def _think(self) -> 'Thought':
        """思考：分析当前状态，决定下一步"""
        # 构建 Prompt
        prompt = self._build_prompt()
        
        # 调用 LLM
        response = self.llm.generate(prompt)
        
        # 解析响应
        thought = self._parse_response(response)
        return thought
    
    def _act(self, action: Action) -> Observation:
        """行动：执行具体操作"""
        if action.type not in self.tools:
            return Observation(
                success=False,
                result="",
                error=f"Unknown action type: {action.type}"
            )
        
        try:
            result = self.tools[action.type](**action.params)
            return Observation(success=True, result=str(result))
        except Exception as e:
            return Observation(success=False, result="", error=str(e))
    
    def _build_prompt(self) -> str:
        """构建思考用的 Prompt"""
        # 学员完成：根据记忆和可用工具构建 Prompt
        pass
    
    def _parse_response(self, response: str) -> 'Thought':
        """解析 LLM 响应"""
        # 学员完成：解析 Thought-Action 格式
        pass

@dataclass
class Thought:
    analysis: str          # 当前分析
    action: Optional[Action]  # 要执行的行动
    is_complete: bool      # 是否完成任务
    final_answer: str      # 最终答案（如果完成）
```

---

#### 工具注册与调用

```python
import requests
import subprocess

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools = {}
        
    def register(self, name: str, func, description: str):
        """注册一个工具"""
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    def get_tool(self, name: str):
        return self.tools.get(name)
    
    def list_tools(self) -> str:
        """列出所有可用工具"""
        result = []
        for name, info in self.tools.items():
            result.append(f"- {name}: {info['description']}")
        return "\n".join(result)

# 注册常用工具
registry = ToolRegistry()

# 搜索工具
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    # 实际实现调用搜索 API
    return f"搜索结果：关于'{query}'的信息..."

registry.register("search", search_web, "搜索网络获取实时信息")

# 代码执行工具
def execute_code(code: str, language: str = "python") -> str:
    """执行代码并返回结果"""
    try:
        if language == "python":
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout or result.stderr
    except Exception as e:
        return f"执行错误：{e}"

registry.register("code", execute_code, "执行 Python 代码")

# 文件读取工具
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()

registry.register("file_read", read_file, "读取文件内容")

# 文件写入工具
def write_file(path: str, content: str) -> str:
    """写入文件内容"""
    with open(path, 'w') as f:
        f.write(content)
    return f"成功写入 {len(content)} 字节到 {path}"

registry.register("file_write", write_file, "写入文件内容")
```

---

#### ✅ 实践项目：基础 Agent 框架

```python
"""
项目：实现一个可工作的基础 Agent
要求：
1. 支持思考 - 行动循环
2. 集成至少 3 个工具（搜索、代码、文件）
3. 能完成简单任务（如"搜索天气并写入文件"）
4. 有最大迭代次数限制
"""

# 学员完成
```

</details>

---

### 7.2 规划能力（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 任务分解

```
复杂任务：帮我分析这家公司的财务状况

分解后：
├── 1. 搜索公司名称和基本信息
├── 2. 获取最新财报数据
├── 3. 计算关键财务指标
│   ├── 3.1 资产负债率
│   ├── 3.2 流动比率
│   └── 3.3 ROE
├── 4. 与行业平均水平对比
└── 5. 生成分析报告
```

#### 思维链（Chain of Thought, CoT）

**CoT Prompt 示例**：
```
问题：小明有 5 个苹果，他给了小红 2 个，又买了 3 个，现在有几个？

普通 Prompt：
"小明现在有几个苹果？"
→ 可能直接回答，容易出错

CoT Prompt:
"小明现在有几个苹果？请逐步思考。"
→ 思考过程：
   1. 初始：5 个苹果
   2. 给小红 2 个：5 - 2 = 3 个
   3. 又买 3 个：3 + 3 = 6 个
   4. 答案：6 个
```

**代码实现**：
```python
class CoTAgent(BaseAgent):
    """支持思维链的 Agent"""
    
    def _build_prompt(self) -> str:
        prompt = """你是一个智能助手。请逐步思考问题，然后决定行动。

可用工具：
""" + self.registry.list_tools() + """

对话历史：
""" + self._format_memory() + """

请按照以下格式思考：

Thought: 分析当前情况和已完成的工作
Plan: 下一步计划
Action: 要执行的动作（如果有）
"""
        return prompt
    
    def _parse_response(self, response: str) -> Thought:
        # 解析 CoT 响应
        lines = response.strip().split('\n')
        
        thought_text = ""
        action = None
        is_complete = False
        final_answer = ""
        
        for line in lines:
            if line.startswith('Thought:'):
                thought_text = line.replace('Thought:', '').strip()
            elif line.startswith('Action:'):
                action = self._parse_action(line)
            elif line.startswith('Final Answer:'):
                is_complete = True
                final_answer = line.replace('Final Answer:', '').strip()
        
        return Thought(
            analysis=thought_text,
            action=action,
            is_complete=is_complete,
            final_answer=final_answer
        )
```

---

#### 思维树（Tree of Thoughts, ToT）

```
ToT 核心思想：探索多种可能的思考路径

        初始问题
           │
    ┌──────┼──────┐
    │      │      │
  路径 1  路径 2  路径 3
    │      │      │
    ▼      ▼      ▼
  评估   评估   评估
    │      │      │
    └──────┼──────┘
           │
      选择最佳路径
           │
           ▼
      继续探索...
```

**代码实现**：
```python
import heapq
from typing import List, Tuple

class ToTNode:
    """思维树节点"""
    def __init__(self, thought: str, score: float, parent=None):
        self.thought = thought
        self.score = score  # 评估分数
        self.parent = parent
        self.children = []
        self.depth = parent.depth + 1 if parent else 0
    
    def __lt__(self, other):
        return self.score > other.score  # 高分优先

class TreeOfThoughtAgent(BaseAgent):
    """支持思维树的 Agent"""
    
    def __init__(self, llm, tools=None, beam_width=3, max_depth=5):
        super().__init__(llm, tools)
        self.beam_width = beam_width  # 束搜索宽度
        self.max_depth = max_depth
    
    def run(self, task: str) -> str:
        # 创建根节点
        root = ToTNode(thought=f"任务：{task}", score=0.0)
        
        # 束搜索
        current_level = [root]
        
        for depth in range(self.max_depth):
            next_level = []
            
            for node in current_level:
                # 生成子节点（多种思考方向）
                children = self._generate_thoughts(node)
                
                # 评估每个子节点
                for child in children:
                    child.score = self._evaluate(child)
                    node.children.append(child)
                    next_level.append(child)
            
            # 保留 top-k 节点
            next_level.sort(reverse=True)
            current_level = next_level[:self.beam_width]
            
            # 检查是否有节点完成任务
            for node in current_level:
                if self._is_complete(node):
                    return self._extract_answer(node)
        
        # 返回最佳节点的答案
        best_node = max(current_level, key=lambda x: x.score)
        return self._extract_answer(best_node)
    
    def _generate_thoughts(self, node: ToTNode) -> List[ToTNode]:
        """生成多种可能的下一步思考"""
        prompt = f"""当前思考：{node.thought}

请生成 3 种不同的下一步思考方向：
"""
        response = self.llm.generate(prompt)
        thoughts = self._parse_thoughts(response)
        
        return [ToTNode(t, score=0.0, parent=node) for t in thoughts]
    
    def _evaluate(self, node: ToTNode) -> float:
        """评估思考节点的质量"""
        prompt = f"""评估以下思考的质量（0-1 分）：
{node.thought}

评分标准：
- 逻辑性：思考是否合理
- 进展性：是否向目标靠近
- 可行性：是否可以执行

分数："""
        response = self.llm.generate(prompt)
        score = float(response.strip())
        return score
```

---

#### ✅ 实践项目：规划模块实现

```python
"""
项目：实现支持 CoT 和 ToT 的规划模块
要求：
1. 实现 CoT 思考链
2. 实现 ToT 束搜索
3. 在复杂任务上测试（如多步骤研究任务）
4. 对比 CoT 和 ToT 的效果
"""

# 学员完成
```

</details>

---

### 7.5 自进化 Agent ⭐（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 反射循环（Reflection Loop）

```
自进化的核心：从经验中学习

执行循环：
┌─────────────────────────────────────────┐
│                                         │
│  1. 执行任务                            │
│     ↓                                   │
│  2. 观察结果（成功/失败）                │
│     ↓                                   │
│  3. 反思分析                            │
│     - 什么做得好？                       │
│     - 什么可以改进？                     │
│     - 学到了什么？                       │
│     ↓                                   │
│  4. 更新策略                            │
│     - 修改 Prompt                        │
│     - 保存新技能                         │
│     - 调整参数                          │
│     ↓                                   │
│  5. 下次任务使用改进后的策略             │
│                                         │
└─────────────────────────────────────────┘
```

#### 反思实现

```python
import json
from datetime import datetime

@dataclass
class Experience:
    """经验记录"""
    task: str
    actions: List[Dict]
    result: str
    success: bool
    reflection: Optional[str] = None
    lesson: Optional[str] = None

class ReflectiveAgent(BaseAgent):
    """支持反思的自进化 Agent"""
    
    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.experiences = []  # 经验库
        self.skills = []       # 技能库
        self.prompt_versions = []  # Prompt 版本历史
        
    def run(self, task: str) -> str:
        # 记录开始
        experience = Experience(
            task=task,
            actions=[],
            result="",
            success=False
        )
        
        # 执行任务
        result = super().run(task)
        experience.result = result
        experience.success = "失败" not in result
        
        # 反思
        reflection = self._reflect(experience)
        experience.reflection = reflection
        
        # 提取教训
        lesson = self._extract_lesson(experience)
        experience.lesson = lesson
        
        # 保存经验
        self.experiences.append(experience)
        
        # 更新技能库
        if experience.success and lesson:
            self._add_skill(task, lesson)
        
        # 更新 Prompt
        self._update_prompt(reflection)
        
        return result
    
    def _reflect(self, experience: Experience) -> str:
        """反思：分析成功或失败的原因"""
        prompt = f"""请反思以下任务的执行过程：

任务：{experience.task}
结果：{experience.result}
成功：{experience.success}

请分析：
1. 什么做得好？
2. 什么可以改进？
3. 如果重来，会采取什么不同的策略？

反思："""
        
        return self.llm.generate(prompt)
    
    def _extract_lesson(self, experience: Experience) -> str:
        """从反思中提取可复用的教训"""
        if not experience.reflection:
            return ""
        
        prompt = f"""从以下反思中提取可复用的经验教训：

{experience.reflection}

用一句话总结核心教训："""
        
        return self.llm.generate(prompt)
    
    def _add_skill(self, task_type: str, skill: str):
        """添加新技能到技能库"""
        self.skills.append({
            "task_type": task_type,
            "skill": skill,
            "created_at": datetime.now().isoformat()
        })
    
    def _update_prompt(self, reflection: str):
        """根据反思更新系统 Prompt"""
        # 记录 Prompt 版本
        self.prompt_versions.append({
            "version": len(self.prompt_versions),
            "reflection": reflection,
            "updated_at": datetime.now().isoformat()
        })
        
        # 实际应用中，这里会更新系统 Prompt
        # 例如添加新的指导原则
    
    def get_similar_experiences(self, task: str, top_k=3) -> List[Experience]:
        """检索相似任务的历史经验"""
        # 简单实现：按任务类型匹配
        # 实际可用向量检索
        return self.experiences[:top_k]
    
    def run_with_experience(self, task: str) -> str:
        """利用历史经验执行任务"""
        # 检索相似经验
        similar = self.get_similar_experiences(task)
        
        if similar:
            # 在 Prompt 中加入历史经验
            self.memory.append({
                "role": "system",
                "content": f"历史经验：{[e.lesson for e in similar if e.lesson]}"
            })
        
        return self.run(task)
```

---

#### 技能库管理

```python
class SkillLibrary:
    """技能库管理"""
    
    def __init__(self, storage_path: str = "skills.json"):
        self.storage_path = storage_path
        self.skills = self._load()
    
    def _load(self) -> List[Dict]:
        """加载技能库"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save(self):
        """保存技能库"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)
    
    def add_skill(self, task_pattern: str, solution: str, tags: List[str] = None):
        """添加新技能"""
        skill = {
            "id": len(self.skills) + 1,
            "task_pattern": task_pattern,
            "solution": solution,
            "tags": tags or [],
            "usage_count": 0,
            "success_rate": 1.0,
            "created_at": datetime.now().isoformat()
        }
        self.skills.append(skill)
        self._save()
    
    def search_skills(self, query: str) -> List[Dict]:
        """搜索相关技能"""
        # 简单关键词匹配
        results = []
        for skill in self.skills:
            if query.lower() in skill["task_pattern"].lower():
                results.append(skill)
        return sorted(results, key=lambda x: x["usage_count"], reverse=True)
    
    def update_skill_stats(self, skill_id: int, success: bool):
        """更新技能使用统计"""
        for skill in self.skills:
            if skill["id"] == skill_id:
                skill["usage_count"] += 1
                # 移动平均更新成功率
                skill["success_rate"] = (
                    skill["success_rate"] * (skill["usage_count"] - 1) + 
                    (1 if success else 0)
                ) / skill["usage_count"]
                self._save()
                break
    
    def get_top_skills(self, limit: int = 10) -> List[Dict]:
        """获取最常用的技能"""
        return sorted(
            self.skills, 
            key=lambda x: x["usage_count"] * x["success_rate"],
            reverse=True
        )[:limit]

# 使用示例
skill_lib = SkillLibrary()

# 添加技能
skill_lib.add_skill(
    task_pattern="搜索并分析",
    solution="1. 先搜索获取信息 2. 提取关键数据 3. 对比分析 4. 生成报告",
    tags=["搜索", "分析", "报告"]
)

# 搜索技能
related = skill_lib.search_skills("分析报告")
print(f"找到 {len(related)} 个相关技能")
```

---

#### ✅ 实践项目：自进化 Agent

```python
"""
项目：实现完整的自进化 Agent
要求：
1. 实现反思循环
2. 实现技能库管理
3. 在多个任务上运行，观察进化效果
4. 可视化成功率提升曲线
"""

# 学员完成
```

</details>

---

### 7.6 多 Agent 协作（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 多 Agent 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     多 Agent 协作系统                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────┐                             │
│                    │  Coordinator│  ← 协调者                    │
│                    │  (管理者)    │                             │
│                    └──────┬──────┘                             │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Researcher │  │   Coder     │  │   Writer    │            │
│  │  (研究员)   │  │  (程序员)   │  │  (写手)     │            │
│  │             │  │             │  │             │            │
│  │ • 搜索信息  │  │ • 编写代码  │  │ • 撰写报告  │            │
│  │ • 收集数据  │  │ • 调试程序  │  │ • 编辑文档  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  协作流程：                                                      │
│  1. Coordinator 接收任务，分解给各 Agent                          │
│  2. 各 Agent 并行执行，共享中间结果                              │
│  3. Coordinator 汇总结果，生成最终输出                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 多 Agent 实现

```python
from enum import Enum

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    REVIEWER = "reviewer"

class MultiAgentSystem:
    """多 Agent 协作系统"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {}
        self.message_queue = []
        
    def add_agent(self, role: AgentRole, agent: BaseAgent):
        """添加 Agent 到系统"""
        self.agents[role] = agent
    
    def run(self, task: str) -> str:
        """执行协作任务"""
        # 1. 协调者分解任务
        coordinator = self.agents[AgentRole.COORDINATOR]
        subtasks = coordinator.decompose_task(task)
        
        # 2. 分配给各 Agent
        results = {}
        for subtask in subtasks:
            agent_role = self._select_agent(subtask)
            agent = self.agents[agent_role]
            results[subtask.id] = agent.run(subtask.description)
        
        # 3. 汇总结果
        final_result = coordinator.synthesize_results(results)
        return final_result
    
    def _select_agent(self, subtask) -> AgentRole:
        """根据子任务选择合适的 Agent"""
        if "搜索" in subtask.description or "研究" in subtask.description:
            return AgentRole.RESEARCHER
        elif "代码" in subtask.description or "编程" in subtask.description:
            return AgentRole.CODER
        elif "报告" in subtask.description or "文章" in subtask.description:
            return AgentRole.WRITER
        else:
            return AgentRole.COORDINATOR

# 专用 Agent 实现
class ResearcherAgent(BaseAgent):
    """研究员 Agent"""
    def __init__(self, llm):
        super().__init__(llm, tools={"search": search_web})

class CoderAgent(BaseAgent):
    """程序员 Agent"""
    def __init__(self, llm):
        super().__init__(llm, tools={"code": execute_code, "file_write": write_file})

class WriterAgent(BaseAgent):
    """写手 Agent"""
    def __init__(self, llm):
        super().__init__(llm, tools={"file_read": read_file, "file_write": write_file})

# 使用示例
system = MultiAgentSystem(llm)
system.add_agent(AgentRole.COORDINATOR, CoordinatorAgent(llm))
system.add_agent(AgentRole.RESEARCHER, ResearcherAgent(llm))
system.add_agent(AgentRole.CODER, CoderAgent(llm))
system.add_agent(AgentRole.WRITER, WriterAgent(llm))

result = system.run("研究 AI 发展趋势，编写代码示例，生成报告")
```

---

#### ✅ 实践项目：多 Agent 协作系统

```python
"""
项目：构建多 Agent 协作系统
要求：
1. 实现至少 3 种角色的 Agent
2. 实现任务分解和结果汇总
3. 完成一个复杂任务（如"研究 + 编码 + 报告"）
4. 分析协作效率
"""

# 学员完成
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 7.1 Agent 基础 | 1 周 | - | 0% | ⏳ |
| 7.2 规划能力 | 2 周 | - | 0% | ⏳ |
| 7.3 工具使用 | 1 周 | - | 0% | ⏳ |
| 7.4 记忆系统 | 1 周 | - | 0% | ⏳ |
| 7.5 自进化 Agent | 2 周 | - | 0% | ⏳ |
| 7.6 多 Agent 协作 | 1 周 | - | 0% | ⏳ |

### 项目清单

- [ ] 基础 Agent 框架
- [ ] CoT/ToT规划模块
- [ ] 工具集成（5+ 工具）
- [ ] 完整记忆系统
- [ ] 自进化 Agent
- [ ] 多 Agent 协作系统

---

> _Agent 是 AI 的终极形态 —— 不仅能思考，还能行动；不仅能执行，还能进化。_
> 
> _—— 悟空_
