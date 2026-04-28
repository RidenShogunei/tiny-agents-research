# tiny-agents-research：Agent 生成新 Agent 的任务委托机制研究

> 生成时间：2026-04-28
> 更新说明：2026-04-28 重写，聚焦核心研究问题
> 关键词：Agent 生成 Agent、任务委托、OpenCode、Qwen3.5-9B、子 Agent 动态生成

---

## 一、核心研究问题

### 1.1 问题陈述

**如何改进 Agent 在生成（spawn）新 Agent 过程中的任务委托方式？**

具体来说：当一个 Agent（母 Agent）需要将任务委托给另一个 Agent（子 Agent）时，
委托的"决策"（选谁、怎么选）和"执行"（怎么传参、怎么同步/异步返回结果）
是否还有优化空间？

### 1.2 相关概念辨析

| 概念 | 说明 | tiny-agents 中的实现 |
|------|------|---------------------|
| **Agent 路由** | 给定任务，选择已有的 Agent 类型处理 | `Router` / `Orchestrator` |
| **Agent 委托** | 当前 Agent 将控制权交给另一个已有 Agent | `delegate` action |
| **Agent 生成（Spawn）** | 当前 Agent 动态创建一个新的 Agent 实例 | ❌ 当前无此机制 |
| **Agent 合成** | 将多个 Agent 的能力组合成一个新的协作单元 | ❌ 当前无此机制 |

**关键区别：** 现有 tiny-agents 的 `delegate` 是**静态路由**——子 Agent 类型是预定义的。
我们关心的是 Agent **自己决定**要不要生成新 Agent、生成什么样的新 Agent。

### 1.3 OpenCode 的启发

OpenCode（基于 Qwen3.5-9B）展示了另一种范式：LLM 本身在一次推理中决定调用哪个工具。
这相当于模型自己在做"动态生成调用"的决策。

类比到 Agent 场景：如果一个 Agent 能够像 OpenCode 调用工具一样，**动态决定**是否生成子 Agent、
生成什么样的子 Agent（指定 prompt/工具/角色），则委托的粒度和灵活性将大幅提升。

---

## 二、当前 tiny-agents 的委托机制

### 2.1 Orchestrator 的 delegate

```python
class AgentOutput(BaseModel):
    action: str        # "respond" | "delegate" | "tool_call" | "review"
    target_agent: str  # 预定义的 agent name，如 "coder", "critic"
    payload: dict      # 传递给目标 agent 的上下文
```

**流程：**
```
Agent A 运行 → 输出 action="delegate", target_agent="B" → 
Orchestrator 切换 current_agent = B → B.run()
```

**问题：**
- `target_agent` 必须是**已注册的 Agent 类型**，无法动态创建
- 传递的只有 `payload`（dict），没有机制指定"新 Agent 的 system prompt / 工具 / 角色"
- 本质是**字符串路由**，不是**行为生成**

### 2.2 BudgetOrchestrator 的固定调度

```python
PHASE1_SCHEDULE = [
    ("reasoner", "REASONER_STEP"),
    ("critic", "CRITIC_STEP"),
    ("reasoner", "REASONER_STEP"),
    ...
]
```

调度规则完全硬编码，Agent 只能按预设顺序执行，无法主动生成新 Agent。

### 2.3 Pipeline 的并行 Step

Pipeline 支持并行执行多个 `PipelineStep`，但：
- Step 是**预先定义**的，不是运行时动态生成的
- 没有"根据任务动态创建新 Step"的机制

### 2.4 总结：当前委托机制的三个局限

| 局限 | 描述 |
|------|------|
| **静态目标** | 只能委托给预定义的 Agent，无法动态生成 |
| **粗粒度传递** | payload 是 dict，无法传递"新 Agent 应该如何行为"的生成指令 |
| **无自我复制** | Agent 无法生成另一个带有自定义配置的新 Agent |

---

## 三、改进方向

### 3.1 动态 Agent 生成（Dynamic Agent Spawning）

允许 Agent 在运行时生成新的 Agent 实例，新 Agent 的配置由母 Agent 的输出决定：

```python
class SpawnOutput(BaseModel):
    action: str = "spawn"
    agent_config: dict  # 新 Agent 的配置
    # agent_config 包含：
    # {
    #   "role": "file_writer",           # 角色描述
    #   "system_prompt": "...",           # 自定义 system prompt
    #   "tools": ["read", "write"],       # 授予的工具有限集合
    #   "model": "Qwen2.5-3B",            # 可选：指定模型
    #   "max_iterations": 5,              # 可选：最大迭代次数
    # }
    task_payload: dict  # 任务输入
```

**这样 Agent 可以自己决定：**
- 是否需要生成专门的子 Agent
- 子 Agent 应该扮演什么角色
- 子 Agent 有哪些工具可用
- 子 Agent 用什么模型

### 3.2 委托决策的透明化

当前 delegate 是"隐式决策"——我们不知道 Agent 为什么选了 B 而不是 C。

改进方案：
- Agent 输出 `delegate` 时，同时输出**决策理由**（`reasoning: "因为任务需要代码审查"`
- 记录每次委托的 reasoning，用于分析优化
- 这与 Qwen3.5-9B 的 thinking 机制天然契合

### 3.3 并行委托 vs 串行委托

**串行（当前）：**
```
母 Agent → [子 Agent B 完成后] → [子 Agent C 完成后] → 母 Agent 汇总
```

**并行（改进）：**
```
母 Agent → [子 Agent B 开始] ← 并行 → [子 Agent C 开始]
          [子 Agent B 和 C 同时运行]
母 Agent 汇总结果
```

Pipeline 的 DAG 拓扑排序已经支持并行，但需要在 Orchestrator 层面增加"并行委托" action。

---

## 四、实验平台配置

### 4.1 Qwen3.5-9B（主力模型）

**配置：**
```bash
CUDA_VISIBLE_DEVICES=3 python3 -m vllm.entrypoints.openai.api_server \
  --model /home/jinxu/.cache/tiny-agents/models/Qwen/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --port 18001 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --load-format safetensors \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**规格：**
| 参数 | 值 |
|------|-----|
| 层数 | 32 |
| hidden_size | 4096 |
| 注意力 | GQA, 4 KV heads |
| max_position | **262,144** |
| bfloat16 显存 | ~18GB |

**注意：** Qwen3.5-9B 默认启用思考机制，无法通过 API 参数关闭。复杂推理任务建议保留思考，简单任务可通过 prompt 约束。

### 4.2 Qwen2.5-3B-Instruct（对比基线）

**配置：**
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
  --model /home/jinxu/.cache/tiny-agents/models/Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen2.5-3B-Instruct \
  --port 18000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 48000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

---

## 五、相关文件索引

| 文件 | 作用 |
|------|------|
| `tiny_agents/core/orchestrator.py` | 串行链式 Orchestrator（当前的 delegate 机制） |
| `tiny_agents/core/agent.py` | BaseAgent 基类（包含 AgentOutput 定义） |
| `tiny_agents/core/pipeline.py` | DAG Pipeline 编排器（并行 Step 的参考实现） |
| `tiny_agents/budget/orchestrator.py` | BudgetOrchestrator（固定调度参考） |
| `tiny_agents/budget/controller.py` | BudgetController（预算决策逻辑参考） |
| `tiny_agents/models/vllm_backend.py` | vLLM 后端封装 |

---

*本文档由 Hermes Agent 自动生成，聚焦 Agent 生成新 Agent 的任务委托机制研究。*
