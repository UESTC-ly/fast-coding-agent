# fast-coding-agent

**Intelligent Coding Agent with Cost-Efficient Routing**

一个生产级的 AI Coding Agent，核心能力是**智能路由**：根据任务复杂度自动选择执行路径，在保证质量的前提下最小化成本。

## 实测结果

24 个任务 × 4 个模型 × 2 种模式 = 192 次运行（`gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna` / `claude-sonnet-5`）。相比全程 Full Agent：

| 模型 | 模型调用节省 | agent loop 节省 | 行为正确率 |
|---|---|---|---|
| gpt-5.6-sol | 27.9% | 57.1% | 48/48 |
| gpt-5.6-terra | 25.4% | 57.0% | 48/48 |
| gpt-5.6-luna | 39.0% | 69.9% | 42/43 |
| claude-sonnet-5 | 45.0% | 74.4% | 44/48 |

只看 FastPath 一次通过的任务，调用次数节省收敛到 **65–68.5%** —— 四个独立模型落在 3.5 个百分点内，反映的是架构常量而非模型能力。三条路径（FastPath 直达、Full Agent 直达、FastPath 放弃后逃逸兜底）在全部四个模型上均验证通过。

完整方法、数据与有效性限制见 **[docs/EXPERIMENT_FASTPATH_ROUTING.md](docs/EXPERIMENT_FASTPATH_ROUTING.md)**。该文档同时记录了跨 provider token 数字不可比的原因，以及结论不可外推至真实大仓库任务的限制。

## 快速开始

```bash
pip install -e .
cp .env.example .env      # 填入你自己的 API key
pytest -q                 # 424 tests

# 复现实验
python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-luna --effort high
```

## 核心特性

### 🧠 智能路由（Intelligent Routing）

```
简单/中等任务 → FastPath (1-3 次调用)
  ├─ 静态特征分析（0 成本）
  ├─ 廉价分类器置信度评估
  └─ 隔离环境快速验证

复杂任务 / FastPath 失败 → Full Agent
  ├─ ReAct 工具循环
  ├─ 上下文分层与压缩
  └─ 子图探索与推理
```

### 🔒 安全隔离（Shadow Workspace）

- 所有修改先在副本中应用和验证
- 三条件同时满足才原子写入真实工作区：
  1. 隐藏验收测试通过
  2. Agent 到达有效完成状态
  3. Diff ⊆ 预期文件集合

### 💰 成本透明（Unified Cost Ledger）

- 唯一计费入口，所有 token 消耗（路由、FastPath 失败、重试、Full Agent）累加进同一账本
- 可离线重放历史任务，对比「全走 Full Agent」vs「Automatic 路由」的成本差异
- 路由轨迹库持续校准置信度阈值，优化成本-质量平衡

## 架构设计

### 1. 模型层（`qqcode.models`）

- **`ModelClient` 协议**：统一接口，抹平 Anthropic / OpenAI 差异
- **Canonical 消息格式**：工具调用、结构化输出、prompt caching 归一化
- **`CostLedger`**：唯一计费入口，分阶段跟踪 token 消耗

### 2. 工作区（`qqcode.workspace`）

- **`Workspace` 协议**：抽象读/写/执行接口
- **Worktree 实现**（M1）：git worktree + 文件副本
- **沙箱后端**（预留）：Docker 容器隔离

### 3. 安全层（`qqcode.safety`）

- **PathGuard**：路径 allowlist，防止逃逸（`..`、symlink、`.git`）
- **CommandGuard**：命令 allow/deny 列表，断网执行
- **WriteQuota**：文件数/行数/字节数限额

### 4. 路由层（`qqcode.routing`）

- **L0 静态特征**：任务文本、文件数、符号定位、仓库规模
- **L1 分类器**：小模型一次调用 → `{tier, confidence, target_files, est_edit_lines, risk_flags}`
- **L2 硬门控**：确定性规则，防止误判

### 5. Agent 层（`qqcode.agents`）

- **FastPath**：单轮或少量迭代，无完整工具循环
- **Full Agent**：LangGraph ReAct，上下文压缩，子图探索

### 6. 记忆层（`qqcode.memory`）

- **会话态**：LangGraph checkpointer（SQLite）
- **项目记忆**：`.qqcode/memory/*.md`，成功后写入，向量检索
- **路由轨迹**：特征 → 决策 → 结果 → 成本，离线校准

## 实现路线图

| 里程碑 | 交付物 | 状态 |
|--------|--------|------|
| M1 骨架 | 项目结构、协议定义、安全管控、`CostLedger` | ✅ 进行中 |
| M2 Provider | Anthropic / OpenAI 适配器，工具调用、缓存、usage | 🔜 |
| M3 FastPath | 路由决策、shadow 应用、隐藏验收、三条件 finalize | 🔜 |
| M4 Full Agent | ReAct 工具循环、上下文管理、artifact store | 🔜 |
| M5 Memory | checkpointer、项目记忆、路由轨迹库、离线校准 | 🔜 |
| M6 CLI + 评测 | 端到端跑通，产出成本对比曲线 | 🔜 |

## 快速开始

```bash
# 安装依赖（需要 Python ≥3.11 和 uv）
uv pip install -e .

# 运行（M1 后可用）
qqcode --task "fix typo in README" --repo /path/to/repo

# 开发
uv pip install -e ".[dev]"
pytest
ruff check .
mypy qqcode
```

## 成本对比示例（M5 后可用）

```
任务集：100 个真实 GitHub issue（已分类难度）

全 Full Agent 模式：
  - 平均成本：$0.42/task
  - 总计：$42.00

Automatic 路由模式：
  - FastPath 命中率：68%
  - FastPath 平均成本：$0.05/task
  - 升级到 Full Agent：32 次
  - 总计：$16.78
  - 节省：60.0%
```

## License

MIT
