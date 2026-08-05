# fast-coding-agent

**一个把「先快后稳」做进架构的 AI Coding Agent。**

大多数编码任务不需要一个反复读文件、试错、再验证的循环就能解决。fast-coding-agent 用双层执行模型抓住这一点：能一次做完的任务走 **FastPath**（单次调用直接产出完整补丁），做不完的自动升级到 **Full Agent**（ReAct 工具循环）。路由判断错了也不会失败 —— FastPath 主动放弃时任务无损地交给 Full Agent 接管。

结果是：调用次数少 25–45%，agent 循环少 57–74%，任务正确率不受影响。

```bash
qqcode --task "修复 auth 模块的 token 过期判断" --repo ./myproject
```

---

## 为什么快

传统 coding agent 对每个任务都付同一份代价：读文件、想、调工具、验证、再想 —— 哪怕任务只是改一行 `a - b` 为 `a + b`。这套循环解决难题必不可少，但用在简单任务上就是纯粹的浪费。

fast-coding-agent 先花极小成本判断任务属于哪一类：

```
                    ┌─────────────────────────────┐
   任务  ─────────▶ │  路由：L0 静态 → L1 分类器   │
                    │        → L2 硬门控          │
                    └──────────┬──────────────────┘
                               │
             ┌─────────────────┴──────────────────┐
             ▼                                    ▼
    ┌────────────────────┐            ┌───────────────────────┐
    │     FastPath       │            │      Full Agent       │
    │  单次调用出补丁    │            │   ReAct 工具循环      │
    │  2 次调用 · 0 循环 │            │   5–8 次调用          │
    └─────────┬──────────┘            └───────────┬───────────┘
              │                                    ▲
              │  主动放弃 / 验收未过               │
              └────────────── 逃逸 ────────────────┘
```

关键设计是 **FastPath 有权说「我做不到」**。它不猜、不硬撑：任务需要探索、或一次改不完，它就返回空补丁并说明原因，路由把任务连同这份诊断交给 Full Agent。所以路由激进一点也安全 —— 押错的代价只是多一次廉价调用，不是任务失败。

---

## 实测佐证

24 个任务 × 4 个模型 × 2 种模式 = **192 次运行**。同一任务集分别跑「全程 Full Agent」和「智能路由」，逐任务配对对比。

**每个模型上 FastPath 都省下了可观的调用与循环：**

| 模型 | 模型调用节省 | agent loop 节省 | 行为正确率 |
|---|---|---|---|
| gpt-5.6-sol | 27.9% | 57.1% | 48/48 |
| gpt-5.6-terra | 25.4% | 57.0% | 48/48 |
| gpt-5.6-luna | 39.0% | 69.9% | 42/43 |
| claude-sonnet-5 | 45.0% | 74.4% | 44/48 |

**隔离到 FastPath 一次做完的任务，节省幅度高度一致：**

| 模型 | 调用节省 | token 节省 |
|---|---|---|
| gpt-5.6-sol | 68.5% | 68.4% |
| gpt-5.6-terra | 65.0% | 73.8% |
| gpt-5.6-luna | 67.4% | 86.0% |
| claude-sonnet-5 | 68.5% | — |

四个独立模型、两家 provider、两种 API 协议，调用节省落在 **65–68.5%** 这 3.5 个百分点的区间内。这种收敛不是偶然：它反映的是架构常量 —— Full Agent 完成一个任务要 5–8 次调用，FastPath 恒定 2 次。模型换了，比例不变。

**三条执行路径全部验证通过**（四个模型均如此）：

| 路径 | 结果 |
|---|---|
| FastPath 直接完成 | 14–18 / 24 任务 |
| Full Agent 直接完成 | 全部任务 |
| FastPath 放弃 → Full Agent 兜底 | 全部逃逸任务最终完成 |

任务集含 8 个 Python 惯用法陷阱（可变默认参数、`if not value` 误判 falsy 值、类属性跨实例共享等），表面合理的改法会被隐藏测试判死 —— 全部通过，说明 agent 在真正理解语义而非模式匹配。

完整数据见 **[docs/EXPERIMENT_FASTPATH_ROUTING.md](docs/EXPERIMENT_FASTPATH_ROUTING.md)**。

---

## 快速开始

```bash
# 需要 Python ≥ 3.11
pip install -e .
cp .env.example .env        # 填入你的 API key

qqcode --task "给 parse_config 加上空文件的处理" --repo ./myproject
```

模式可以手动指定：

```bash
qqcode --task "..." --repo ./p --mode auto    # 默认：智能路由
qqcode --task "..." --repo ./p --mode fast    # 只用 FastPath，不升级
qqcode --task "..." --repo ./p --mode full    # 直接用 Full Agent
```

作为库嵌入：

```python
from qqcode.config import Config
from qqcode.orchestrator import run_task

result = run_task(
    "修复 add() 的符号错误",
    repo=Path("./myproject"),
    config=Config.from_env(),
    mode="auto",
)
print(result.mode_used, result.turns_used, result.ledger.summary())
```

---

## 安全隔离

所有改动先落在 **shadow workspace**（git worktree 副本）里，三个条件同时满足才原子写回真实仓库：

1. **隐藏验收测试通过** —— 由任务方提供，agent 看不到
2. **Agent 到达有效完成状态** —— 不是超轮数、不是卡死、不是截断
3. **Diff ⊆ 预期文件集** —— 改动范围超出声明即拒绝

任何一条不满足，真实仓库一个字节都不会变。

三道防线在写入前生效：

| 防线 | 作用 |
|---|---|
| `PathGuard` | 路径 allowlist，拦 `..`、symlink、`.git` 逃逸 |
| `CommandGuard` | 命令 allow/deny 列表，执行时断网 |
| `WriteQuota` | 文件数 / 行数 / 字节数限额 |

FastPath 补丁在写入前还有一道防篡改检查：任何试图写入验收目录的补丁直接拒绝，防止在测试收集阶段执行任意代码。

---

## 成本透明

`CostLedger` 是唯一计费入口 —— 路由判断、失败的 FastPath、provider 重试、子代理、升级后的 Full Agent，全部计入同一账本。没有任何绕过计费的出网路径。

```python
result.ledger.summary()
# {'calls': 2, 'automatic_total': 1019,
#  'by_phase': {'routing': 265, 'fastpath': 754, 'fullagent': 0, 'subagent': 0}}
```

每次运行写一条 trace 到 `.qqcode/trace.db`：路由特征、决策、结果、分阶段成本。可以离线重放整个任务集，对比不同置信度阈值下的成本-质量曲线，不花一分钱 token。

---

## 架构

```
qqcode/
├── models/        Provider 适配层（Anthropic / OpenAI）
├── routing/       L0 静态 → L1 分类器 → L2 门控 + FastPath 执行
├── agents/        Full Agent（LangGraph ReAct）+ 子代理
├── workspace/     Shadow workspace（git worktree）
├── safety/        PathGuard / CommandGuard / WriteQuota
├── acceptance/    隐藏验收测试注入与清理
├── memory/        trace 库 + 离线重放 + 项目记忆
├── tools/         工具注册表与执行器
└── skills/        任务相关的技能注入
```

**Provider 无关**：`ModelClient` 协议抹平两家差异 —— 工具调用格式、结构化输出、prompt caching、usage 口径全部归一化到统一形状。换 provider 不改上层任何代码。

**分层路由**：L0 用静态特征（任务文本、文件数、符号定位、仓库规模）零成本筛掉明显的极端情况；L1 用小模型一次调用给出 `{decision, confidence, files, reasoning}`；L2 是确定性硬门控，用规则否决 L1 的高风险误判。

---

## 开发

```bash
pip install -e ".[dev]"

pytest -q              # 424 tests
ruff check .
mypy qqcode
```

复现实验：

```bash
# 单模型，24 任务 × 2 模式
python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-luna --effort high

# 四模型矩阵
python benchmarks/trivial_paths_benchmark.py \
  --models gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna,claude-sonnet-5 --effort high
```

benchmark 自带 fixture 自检：每个隐藏测试必须在原始代码上失败、在正确修复后通过，否则拒绝运行。

---

## License

MIT
