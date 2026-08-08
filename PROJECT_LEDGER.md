# QQCode 项目台账

> **维护约定**：每个里程碑完成、范围变更或关键决策后立即更新本文件。
> 最后更新：2026-08-07 · v1.0.0 已发布 · 660 tests + 2 xfailed（strict）· ruff/mypy clean

---

## 一、最终目标

构建一个**面向真实编程任务的生产级 Coding Agent**，用智能路由把大多数任务的成本压到全 Full Agent 模式的一半以下，同时不牺牲成功率。

三条可测量的**目标指标**（2026-08-06 定调变更：由硬性验收门槛放宽为目标）：

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 成本节省 | Automatic 模式总成本 ≤ 全 Full Agent 的 50% | 同一任务集离线重放对比 |
| 成功率 | Automatic ≥ 全 Full Agent 的 95% | 三条件验收通过率 |
| 安全性 | 零非预期修改逃逸 | snapshot 全量 diff 比对 |

**这三个数字是目标，不是通过/不通过的门槛。** 达不到时忠实记录实测值与未达原因，不调整测量方式去迁就目标，也不把「未测量」写成「已达标」。指标的用途是指示方向和暴露差距；一个诚实的 13.8% 比一个粉饰过的「达标」有用得多。

**成本口径**：FastPath 前置请求、失败请求、Provider 重试、子代理消耗、升级后的 Full Agent 消耗，全部计入 `automatic_total`。不允许任何绕过计费入口的出网路径。

### 1.1 指标达成实情（2026-08-07 核对，数据源已逐字核实）

数据源：`benchmarks/results/qqcode-abba-20260807-063237/report.json`（AB/BA，openai/gpt-5.6-terra，5 个 derivable fixture，automatic 8 runs / full 9 runs，`cycles: 1`）。

| 指标 | 目标 | 实测 | 如实记录 |
|------|------|------|------|
| 成本节省 | ≤ 50% | **+18.9%**（8 个可比对）/ **+4.0%**（报告口径：两者皆成功的 4 对） | 均未达目标。均值口径：automatic 30,625 vs full 43,080 tokens/run |
| 成功率 | ≥ 95% | **automatic 6/8 = 75%**，full 4/9 = 44% | 未达目标，但 automatic **高于** full |
| 安全性 | 零逃逸 | — | **这批数据完全没测**，见下方仪器缺陷 |

**仪器缺陷（决定了上面每一个数怎么读）**：`benchmarks/qqcode_benchmark.py:698` 的 `run_task(...)` 调用**不传 `harness=`**（已逐行核实：该调用只有 task/repo/config/mode/dry_run/provider/model/reasoning_effort/trace_store）。所以：

- 三条件里的**条件 1（隐藏测试通过）在整个评测里恒为空**。`behavioral_complete` 不是三条件收敛的产物，是跑完之后由 `_run_acceptance` 另外打 `test_patch` 评出来的。
- 上表的成功率**不是三条件通过率**。台账早前把 v1.0.0 那次也笼统称作「三条件」，是不准确的。
- 零逃逸在这批数据里无从谈起。旁证：本轮新增的信任警告 17 次运行触发 0 次，说明 harness 通道确实一次都没走过（`_harness_ran` 在 benchmark 里只用于**检测输出里的警告**，不注入 harness）。

**FastPath 自身命中率：2/6 = 33%**（`fastpath.attempted: 6, completed: 2, upgrades: 6`）。值得记下的是：即便 FastPath 只有三分之一命中，**automatic（FastPath + 升级）在成功率和 token 上同时优于纯 full**（75% vs 44%，30,625 vs 43,080）。这是「双档 + 路由」这个产品形态目前唯一的正面证据，但它出自上述残缺的门，且 n 很小，不能当结论用。

**报告分母不一致（既有缺陷，非本轮引入）**：`_build_report` 的 `measurable`（第 955 行）只从 `clean_auto` 取，所以没有 automatic 运行的 `full_only` fixture 从可测量计数里消失，而它的 full 运行仍留在 `behavioral_rate_measurable_full` 的分母里。报告显示 `measurable_tasks: 4`，实际跑的是 5 个。两个 measurable 率因此不可比。

**先前那个 13.8%** 来自 v1.0.0 的 FastPath 预取修复 A/B（3 个单文件任务，arm A 用 monkeypatch 还原修复前行为）。它和上面的 +18.9% **不是同一个数的两个版本，是两次不同的测量**：前者 n=3、只比预取修复前后；后者 n=5 fixture / 17 runs、比 automatic 与 full。以后者为主，前者降为参考。

**为什么分母是 5 不是 15**：2026-08-06 的离线审计确认 15 个 fixture 里只有 5 个的隐藏断言能从任务陈述推导（见 §7 R9），在不可推导的 fixture 上一个正确修复也得 0 分。

---

## 二、产品形态

**QQCode** = CLI 工具 + 可嵌入的 Python 库。

```
qqcode --task "修复 auth 模块的 token 过期判断" --repo ./myproject
```

### 核心差异点

1. **智能路由（Automatic 模式）**
   - 边界清晰的简单/中等任务 → **FastPath**：1–3 次模型调用，隔离验收通过后原子写入
   - FastPath 证据不足 / 补丁不可用 / 验收失败 / 越界 → 从**干净工作区**升级 Full Agent
   - 升级时丢弃 shadow 回到基线，但把结构化失败诊断传给 Full Agent

2. **三条件收敛**：最终成功必须同时满足
   - 隐藏行为验收通过（外部注入，Agent 不可见）
   - Agent 完成状态有效
   - 无非预期修改（diff ⊆ 预期文件集）

3. **成本全透明**：唯一计费入口，分阶段账本，可离线重放校准

4. **子代理工作模式**：内置 7 个预设，Full Agent 按需 spawn，只回收结论不污染主上下文

### 用户可见的模式

| 模式 | 行为 |
|------|------|
| `--mode auto`（默认） | 智能路由，FastPath 优先，失败自动升级 |
| `--mode fast` | 强制 FastPath，失败即报错不升级 |
| `--mode full` | 跳过路由直接 Full Agent |
| `--dry-run` | 只在 shadow 中执行，输出 diff 不 finalize |

### 内置子代理预设

| 预设 | 模型档 | 隔离级别 | 用途 |
|------|--------|----------|------|
| `explorer` | fast | 只读 | 定位实现、理解调用链 → 结构化代码地图 |
| `reviewer` | balanced | 只读 | 代码缺陷审查 → 分级 findings |
| `security-auditor` | deep | 只读 | 安全专项：注入、认证、密钥、路径遍历 |
| `planner` | deep | 只读 | 有序实现计划 + 风险 + 验证方式 |
| `test-writer` | balanced | shadow 可写 | 补测试，可执行验证 |
| `build-fixer` | fast | shadow 可写 | 修构建/类型/lint 错误，最小 diff |
| `doc-writer` | fast | shadow 可写 | 撰写更新文档 |

用户可通过 `register_preset` 注册自定义模式，或 `spec.derive(...)` 派生变体。子代理默认不继承任何 MCP 服务器与 skill，需在 spec 里用 `mcp_servers` / `pinned_skills` 显式授予。

### 工具 / MCP / Skills 分层门控

| 表面 | 内置工具 | MCP | Skills | 计费阶段 |
|------|----------|-----|--------|----------|
| FastPath | 最小集 | **完全不可见** | 仅单个 pinned（`fastpath_safe`） | `fastpath` |
| Full Agent | 全量 | opt-in + allowlist，默认关 | 索引常驻 + 按需加载正文 | `fullagent` |
| 子代理 | 按 `allowed_tools` | 默认不继承，需显式授予 | 仅 spec 里 pin 的 | `subagent` |

四者定位区分：**function calling 与 tool use 是同一件事的两个名字**（OpenAI / Anthropic 各自的叫法）；**MCP 是工具的分发协议**；**skill 是知识而非能力**——无副作用，不需要模型「调用」，只需在合适时机进入上下文，做成 tool 会白白多一轮往返。

---

## 三、技术选型（已确认）

| 维度 | 决定 | 理由 |
|------|------|------|
| 技术栈 | Python 3.11+ / LangGraph | checkpointer、子图、streaming 生态最完整 |
| 模型层 | 自研薄适配层，直连官方 SDK | 保留 cache 断点控制与计费保真度 |
| 隔离 | git worktree / 文件副本 | 无外部依赖，Workspace 接口预留 Docker 后端 |
| 验收 | 外部注入的隐藏测试 | Agent 全程不可见，防止针对性作弊 |
| MCP 定位 | Full Agent 的可选扩展，默认全关 | schema 体积（单 server 6–9k tokens）会直接抹平 FastPath 的成本优势 |
| Skills 优先级 | 高于 MCP | 仓库约定是 agent 最常犯错的地方，skill 直接提升成功率；MCP 只是生态复用红利 |

---

## 四、主线任务

| 里程碑 | 交付物 | 状态 | 验证方式 |
|--------|--------|------|----------|
| **M1 骨架** | 项目结构、canonical 协议、`CostLedger`、`Workspace` 抽象 + worktree 实现、安全管控、子代理规格层 | **完成** | 全绿：pytest / ruff / mypy strict |
| **M1.5 工具与 Skill 契约层** | `ToolRegistry`、artifact 压缩、`MCPServerConfig` 准入规则、`SkillIndex`、子代理 MCP/skill 授权字段 | **完成** | 222 tests · ruff clean · mypy strict clean |
| **M2 Provider 适配层** | Anthropic / OpenAI 双适配器：工具调用、结构化输出、cache 断点、usage 归一化、唯一计费入口 `BilledClient` | **完成** | 67 条契约+计费测试 · mypy strict clean |
| **M3 FastPath + 路由** | L0 静态特征、L1 分类器、L2 硬门控、shadow 应用、隐藏验收注入、三条件 finalize；接入 skill 路由信号；`.env` 配置层 | **完成** | 319 tests · ruff clean · mypy strict clean · Anthropic smoke 5/5 |
| **M4 Full Agent + 子代理执行** | 10 个内置工具 + executor、ReAct 循环、5 个终止条件、artifact 压缩接线、spawn 回调、FastPath 升级上下文传递 | **完成** | 350 tests · ruff clean · mypy strict clean |
| **M4.5 MCP 客户端** | stdio / SSE 连接、懒启动、崩溃隔离、写类服务器约束到 shadow 根 | **完成** | stdio transport 完成；SSE 预留接口；15 条测试全绿 |
| **M5 Memory + 轨迹库** | checkpointer、`.qqcode/memory`、路由轨迹记录、离线重放校准 τ/L/K；量化 skill 对 FastPath 命中率的提升 | **完成** | 30 条测试；`qqcode trace replay` 输出 τ/L/K 三维校准表 + skill 影响列表 |
| **M6 CLI + 评测** | `orchestrator.run_task`（三模式 + 升级路径 + finalize 门控）、typer CLI、rich 输出、真实 API 端到端验证 | **完成** | Anthropic auto/fast/full 三模式端到端通过；finalize 原子写回验证；无 staging/backup 残留 |
| **M7 会话交互层**（原规划外新增） | `qqcode --chat` REPL、`.qqcode/sessions.db` 会话持久化 + `--resume`/`--continue`、快照式 `/undo`、实时工具调用输出、shadow 从工作树 seed、脏仓库守卫 | **完成** | 见 `docs/DESIGN_CONVERSATIONAL_LAYER.md`；601 tests |
| **v1.0.0 发布** | 版本号 0.1.0 → 1.0.0，tag + GitHub Release；CLI 表面 / `run_task` 签名 / trace schema 纳入兼容性承诺 | **完成** | `github.com/UESTC-ly/fast-coding-agent/releases/tag/v1.0.0`；已知边界在 release notes 中明示 |

---

## 五、当前进展

### M1 骨架（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| 模型协议 | `qqcode/models/protocol.py` | `Msg` canonical 格式、`ToolSpec`、`OutputSpec`、`ModelClient` 协议、`CostLedger` |
| 工作区协议 | `qqcode/workspace/protocol.py` | `Workspace` 协议、`FileSnapshot`、`snapshot_directory` |
| 工作区实现 | `qqcode/workspace/worktree.py` | `WorktreeWorkspace`：git worktree / 副本回退、原子 finalize、密钥剥离 |
| 安全管控 | `qqcode/safety/guards.py` | `PathGuard`（逃逸防护）、`CommandGuard`（allow/deny）、`WriteQuota` |
| 子代理规格 | `qqcode/agents/subagent.py` | `SubAgentSpec`/`SubAgentResult`、7 个内置预设、注册表 |

**已验证的不变量**：
- 写入不泄漏到源仓库（无 finalize 则源仓库零变更）
- symlink 逃逸、`..` 穿越、`.git/`、`.env` 全部被拒
- 密钥环境变量不进子进程（`ANTHROPIC_API_KEY` → `ABSENT`）
- snapshot 能检出修改 / 新增 / 删除三类差异
- FastPath 失败成本在升级后仍计入 `automatic_total`
- finalize 原子性：失败自动回滚，不留 staging 残留
- 只读子代理规格无法被授予写工具（构造期即拒绝）

### M4.5 MCP 客户端（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| MCP 客户端 | `qqcode/tools/mcp_client.py` | `MCPClient`：懒启动、stdio/SSE transport、崩溃隔离、shadow root 约束 |
| 工具执行器扩展 | `qqcode/tools/executor.py` | `ToolExecutor._call_mcp_tool()`：解析 `mcp__<server>__<tool>` 命名空间并路由 |
| 测试 | `tests/test_mcp_client.py` | 15 条测试：启动、工具调用、崩溃检测、shadow root、shutdown |

**已验证的不变量**：
- MCP 服务器在 `register_server()` 前不启动（懒加载）
- 写类服务器收到 `--root <shadow_root>` 参数，文件操作被限制在 shadow workspace 内
- 服务器崩溃被检测并作为工具错误返回，不导致 agent 进程退出
- `tool_allowlist` 正确过滤服务器暴露的工具
- `shutdown_all()` 优雅终止 stdio 进程，超时后强制 kill
- SSE transport 保留接口但抛出 `NotImplementedError`



| 模块 | 文件 | 内容 |
|------|------|------|
| 工具注册表 | `qqcode/tools/registry.py` | `ToolRegistry`（schema 单一来源）、`RegisteredTool`、MCP 命名空间、分层可见性 |
| 结果压缩 | `qqcode/tools/artifacts.py` | `ArtifactStore` 协议 + 内存实现、`ResultPolicy`、`build_tool_result` |
| MCP 配置 | `qqcode/tools/mcp.py` | `MCPServerConfig` 准入校验、`MCPCapability`、shadow 根约束 |
| Skill 定义 | `qqcode/skills/skill.py` | `Skill`、frontmatter 解析、glob/keyword 匹配、`RoutingHint` |
| Skill 索引 | `qqcode/skills/index.py` | `SkillIndex`：发现、匹配、分层选择、FastPath 单 pin |
| 子代理授权 | `qqcode/agents/subagent.py` | 新增 `mcp_servers` / `pinned_skills`，均默认为空 |

**已验证的不变量**：
- MCP 工具强制 `mcp__<server>__<tool>` 命名，第三方 `read_file` 无法劫持内置工具
- MCP 在 FastPath 上不可见，即使显式 enable 也被过滤；注册期即拒绝 `fastpath` tier
- 写类 MCP 服务器未声明 `shadow_root_arg` 时拒绝准入；`launch_command` 自动附加 shadow 根
- 网络类 MCP 服务器标记为非 replay-safe，不进离线重放数据集
- 结构化输出与真实工具互斥，同时请求即抛错而非静默丢弃
- 超限工具结果压缩为头尾摘要 + artifact id，全文可从 store 取回
- `ResultPolicy` 拒绝「压缩后仍超预算」的参数组合
- FastPath 无常驻 skill 索引；仅当唯一 `fastpath_safe` skill 命中时注入单个正文
- 子代理默认不继承 MCP 与 skill，需显式授予
- skill 声明 `fastpath_safe` 与 `routing_hint: full` 冲突时构造期即拒绝

### M2 Provider 适配层（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| 错误分类 | `qqcode/models/errors.py` | `ProviderError`（status 驱动的 retryable 判定）、`BudgetExhaustedError` |
| Anthropic 适配器 | `qqcode/models/anthropic_adapter.py` | `AnthropicAdapter`、content block 转换、`cache_control` 断点注入、强制工具调用、usage 归一化 |
| OpenAI 适配器 | `qqcode/models/openai_adapter.py` | `OpenAIAdapter`、tool_calls 转换、`json_schema` strict 模式、usage 归一化（减去缓存部分） |
| 唯一计费入口 | `qqcode/models/billing.py` | `BilledClient`（retry + ledger 包装）、`RetryPolicy`（指数退避） |
| 模型层协议补充 | `qqcode/models/protocol.py` | `ModelTier` 迁入、`CostLedger.retried_calls` |

**已验证的关键不变量**（`tests/test_adapters.py` 43 条 + `tests/test_billing.py` 24 条，duck-typed fake SDK，无需 API key）：
- Anthropic `input_tokens` 不含缓存；OpenAI `prompt_tokens` 含缓存 → 归一化后口径一致（**计费保真度修正**）
- 同一工作量两 provider 报告相同的 `(input_tokens, output_tokens, cache_read_tokens)`
- OpenAI `cached > prompt` 的异常报告不产生负数（`max(0, prompt - cached)`）
- 结构化输出与真实工具互斥，两 adapter 同时请求时均 raise `ValueError`，且不计费
- 429/5xx 可重试，其余立即失败；每次重试的 token 计入 ledger（`retried=True`）
- `BudgetExhaustedError` 拦截发生在发送前，adapter 的 `invoke` 未被调用
- 两 adapter 的 `invoke` 返回相同 `Completion` 结构，stop_reason 映射到 canonical 名称
- SDK 异常保留 `status_code`，无状态码的异常不可重试（防止无限重试未知错误）

### M3 FastPath + 路由（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| 三层路由 | `qqcode/routing/router.py` | `route_task`：L0 静态特征 → L1 廉价分类器 → L2 确定性硬门控；`RoutingDecision` / `RoutingResult` |
| FastPath 执行 | `qqcode/routing/fastpath.py` | `execute_fastpath`：单次强制结构化输出生成整文件补丁 → shadow 写入 → 隐藏验收 → 三条件校验 |
| 配置层 | `qqcode/config.py` | `Config.from_env`：`.env` 解析（内联实现，不引入 python-dotenv）、双 provider key + base_url |
| 配置模板 | `.env.example` | `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DEFAULT_PROVIDER` |
| skill 导出修复 | `qqcode/skills/__init__.py` | 显式 `__all__` 重导出 `RoutingHint` / `Skill` / `SkillIndex` / `load_skill` |

**路由阈值（初期保守值，待 M5 轨迹库校准）**：

| 参数 | 值 | 作用 |
|------|-----|------|
| `MAX_FASTPATH_TASK_LENGTH` | 500 字符 | L0：任务描述过长即升级 |
| `MAX_FASTPATH_FILES` | 3 | L2：L1 预测文件数超限即覆盖为 FullAgent |
| L2 置信度下限 | 0.7 | L2：L1 说 FastPath 但置信度不足即升级 |
| `FULLMUST_KEYWORDS` | refactor / architecture / migrate / redesign / investigate / … | L0：命中即强制 FullAgent |

**已验证的不变量**（9 条测试，`tests/test_routing.py`）：
- L0 命中复杂关键词 / 任务超长 → 强制 FullAgent，置信度 1.0，零模型调用
- L0 读取 skill 的 `routing_hint`：`FULL` 强制升级，`FAST` 给 0.85 置信度走 FastPath
- L1 分类器以 `phase="routing"` + `ModelTier.FAST` 计费，输出走强制 `classify_task` 工具调用
- **L2 硬门控能覆盖 L1**：L1 说 FastPath 但预测 4 个文件 → 覆盖为 FullAgent
- **L2 硬门控能覆盖 L1**：L1 说 FastPath 但置信度 0.6 < 0.7 → 覆盖为 FullAgent
- 无 client 或 L1 抛错 → fallback 到 FastPath（置信度 0.5），不阻塞任务

**FastPath 的五个升级出口**（`escalation_reason`，均携带结构化 `diagnostic` 传给 Full Agent）：
`model_error` · `no_patch` · `truncated` · `declined` · `write_error` · `unexpected_modifications` · `acceptance_tampering` · `acceptance_failed` · `harness_error`

**已修正的关键缺陷**（运行时发现，影响成本前提）：

> **FastPath 对已存在文件的修改必然 declined。** 原始实现要求模型输出完整文件内容，却不在 prompt 里提供任何文件内容。模型每次正确地拒绝："盲写整文件会丢失现有代码"。L1 路由判断正确（conf=0.97），是 FastPath 自身的缺陷：Automatic 模式 = 路由 + FastPath + 升级 > 单跑 Full Agent，比基线更贵。
>
> **修法**：`execute_fastpath` 在组装 prompt 前调用 `_prefetch_files(files_hint, workspace)` 预取文件内容（本地读取，零额外调用），内联进用户消息的「Current file contents」区块。上限 `MAX_PREFETCH_TOTAL_CHARS=20k` / 单文件 `MAX_PREFETCH_FILE_CHARS=8k`，超限任务交给 Full Agent 选择性读取。
>
> 修复后验证：同一"add docstring"任务，auto 模式 FastPath 一次命中（routing 345 + fastpath 505 = **850 tokens**，无升级）。

**已验证的不变量**（9 条测试 + 真实 API 端到端）：
- L0 命中复杂关键词 / 任务超长 → 强制 FullAgent，置信度 1.0，零模型调用
- L0 读取 skill 的 `routing_hint`：`FULL` 强制升级，`FAST` 给 0.85 置信度走 FastPath
- L1 分类器以 `phase="routing"` + `ModelTier.FAST` 计费，输出走强制 `classify_task` 工具调用
- **L2 硬门控能覆盖 L1**：L1 说 FastPath 但预测 4 个文件 → 覆盖为 FullAgent
- **L2 硬门控能覆盖 L1**：L1 说 FastPath 但置信度 0.6 < 0.7 → 覆盖为 FullAgent
- 无 client 或 L1 抛错 → fallback 到 FastPath（置信度 0.5），不阻塞任务
- FastPath declined 时升级到 Full Agent，FastPath 的 token 计入 `automatic_total`（唯一计费入口成立）
- `AcceptanceHarness` 注入测试后清理干净，不影响 diff 检查
- agent 往 `.qqcode_acceptance/` 写 conftest.py 的篡改尝试在 write 前被拒（`ACCEPTANCE_TAMPERING`）

### M4 Full Agent + 子代理执行（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| 内置工具 schema | `qqcode/tools/builtins.py` | 10 个工具定义 + `default_registry()`；所有工具 tier 不含 `fastpath`（否则破坏结构化输出互斥） |
| 工具执行器 | `qqcode/tools/executor.py` | `ToolExecutor`：guard 执行、artifact 压缩、spawn 回调；10 个工具处理器 |
| Full Agent 循环 | `qqcode/agents/full_agent.py` | `execute_full_agent`：ReAct 循环 + 5 个终止条件 |
| adapter factory | `qqcode/models/factory.py` | `build_client` / `build_adapter` / `uniform_tiers` |

**内置工具列表**（`fullagent` + `subagent` 两个 tier）：
`read_file` · `list_files` · `grep` · `write_file` · `edit_file` · `run_command` · `read_artifact` · `read_skill` · `finish` · `spawn_subagent`（仅 `fullagent`）

**5 个终止条件**（全部有离线测试锁定）：

| 条件 | 触发 |
|------|------|
| `explicit` | agent 调用 `finish` 工具（条件 2 满足，valid finish state） |
| `max_turns` | 达到 `max_turns` 上限（默认 30） |
| `budget` | `BudgetExhaustedError` 在发送前拦截 |
| `stuck` | 连续 2 轮无工具调用，或同一错误重复 3 次 |
| `error` | provider 不可重试错误（非 4xx/5xx retryable） |

**已验证的不变量**（21 条 executor 测试 + 10 条 loop 测试）：
- 路径穿越（`../../etc/passwd`）被 `PathGuard` 拒，返回 error 结果而非抛异常
- `.env` 写入被拒
- `edit_file` 要求 `old_string` 唯一；出现 0 次或 N>1 次均返回错误
- `run_command` 以数组传参；`curl` / `wget` 等网络工具被拒
- 超限工具结果压缩为摘要 + artifact id；全文可通过 `read_artifact` 取回
- 无 spawn callback 时 `spawn_subagent` 返回 error（不 raise）
- FastPath 升级上下文注入 system prompt，Full Agent 获知前次失败原因
- `tool_registry=None` 时自动使用 `default_registry()`
- stuck 检测：`.env` 写入失败 3 次连续相同错误 → 退出循环

### M6 CLI + 评测（已完成）

| 模块 | 文件 | 内容 |
|------|------|------|
| 编排器 | `qqcode/orchestrator.py` | `run_task`：三模式（auto/fast/full）、FastPath 升级路径（丢弃 shadow 建新工作区）、Full Agent 验收与 finalize 门控 |
| CLI | `qqcode/cli.py` | typer 单命令入口；rich 输出：成功/失败状态、改动文件清单、分阶段 token 表；exit code 0/1/2 |

**安装后用法**（`pyproject.toml` 已有 `qqcode = "qqcode.cli:main"`）：
```
qqcode --task "..."  --repo ./myproject  [--mode auto|fast|full]  [--dry-run]
       [--provider anthropic|openai]  [--model MODEL_ID]  [--max-turns N]
```

**真实 API 端到端验证结果**（`claude-sonnet-5`）：

| 场景 | 结果 | token 成本 |
|------|------|-----------|
| auto → FastPath 命中 | ✅ fastpath · dry run | routing 345 + fastpath 505 = **850** |
| auto → FastPath 升级 Full Agent | ✅ fullagent · committed | routing 125 + fastpath 186 + fullagent 509 = **820** |
| full → Full Agent 直接写回 | ✅ fullagent · committed | fullagent 277（无路由/FastPath 开销） |
| dry-run 后源文件不变 | ✅ | — |
| finalize 原子写回 | ✅ 无 staging/backup 残留 | — |
| 升级路径计费完整性 | ✅ FastPath 失败的 186 tokens 计入 `automatic_total` | — |

### 预取定位器（2026-08-07，`0f50c4b`）

**要解决的浪费**：任务不含文件名、无 `files_hint`、L1 恢复也失败时，FastPath 仍然会花一次调用。预取读不到任何文件，prompt 里没有代码，模型只能走文档化的出口。实测 4/4 全 declined，每次 23k–44k tokens；同一任务有线索时只需 5.8k。

| 模块 | 文件 | 内容 |
|------|------|------|
| 词法定位器 | `qqcode/routing/locate.py` | `locate_files`：IDF 排序 + 文件名词干前缀 + 测试文件降权；有界目录遍历 |
| 预取接线 | `qqcode/routing/fastpath.py` | `_resolve_advisory_or_locate`：建议 hint 能验证则用它，否则用定位器 |
| 测试 | `tests/test_locate.py` | 32 条（`76d4fa5` 补了 2 条 prompt 级接线测试） |

**生产路径实测 recall@3**（5 条 derivable 语句，cap=3，无 `files_hint` 无 `prefetch_hint`，即上述 4/4 必败的形状）：

| 方法 | recall@3 |
|------|------|
| **定位器** | **3/5（60%）**，单次 6–22ms |
| `_PATH_TOKEN` 文本提取 | 0/5 |
| L1 分类器猜测 | 1/4 |

两个 miss 是结构性的：`importlib double import` 的语句与 `pathlib.py` 词汇无交集；`dynamic xfail` 把 `nodes.py`/`python.py` 排在 `skipping.py` 前。

**两个被实测否决的机制**（记在模块 docstring 里防止重提）：定义点加权 @3 无变化、@5 更差（测试文件大量 `def test_*`，它成了测试文件放大器）；标识符精确匹配把 @3 打到 20%。

**一处自我纠正**：最初那个 60% 用的停用词表含 `regression`/`missing`/`twice` 等缺陷词。用另外 14 条语句做留一法后，增益消失——**那张表是照着答案拟合的**。改成只含功能词重建，60% 通过**测试文件降权**重现，而早前笔记把降权记作「@3 无贡献」，那是拟合表压住散文词造成的假象。

**成本边界（实测）**：读+排序吞吐约 70MB/s，220 文件 / 2.3MB 约 50ms。本仓库原始树有约 6 万个被 gitignore 的 `.py`（`benchmarks/results/` 下的仓库副本），触发 2000 文件上限，115ms 后返回空。生产用 `use_git=True`，worktree 只有 71 个被跟踪的 `.py`，上限只是兜底。

**已验证的不变量**（变异测试 20/22，8 个接线类变异全杀）：
- 定位器结果只进 `prefetch_hint`，永不进 `files_hint`（后者兼任条件 3 契约，猜错是拒绝正确补丁而非浪费 token）
- 建议 hint 能验证时优先于定位器
- hint 指向不存在的文件时定位器不顶替它（那是「创建」语义）
- 测试文件降权而非排除；词干前缀下限 5 字符；树超上限返回空而非截断排名
- **定位到的文件内容真的进了 prompt**（`76d4fa5`，经 `run_task` 断言，非 `resolve_prefetch_paths` 层）
- 两个存活变异是构造上冗余，测试里已写明，不假装覆盖

**变异 harness 自身的两个缺陷（2026-08-08 实测修复，`/tmp/qq_mutate_locate.py`）**。两者都会把「没测到」伪装成结果，方向都是危险的那一侧：

| 缺陷 | 机制 | 后果 | 修法 |
|------|------|------|------|
| 陈旧字节码 | CPython 按 (源文件 mtime 整秒, 字节数) 校验 `.pyc`。`UBIQUITY_CUTOFF = 0.5` → `1.1` 字节数不变（13439 → 13439），改写落在同一秒内时旧 `.pyc` 被判为有效 | 变异成为空操作，却报告 **SURVIVED**（诬告测试无效）。实测 3 次试验中 2 次复现：磁盘上是 `1.1`，新进程 `import` 看到 `0.5` | 清 `__pycache__` + `PYTHONDONTWRITEBYTECODE=1`；并对 `NAME = 字面量` 类变异在子进程中核实新值可见，不可见则报 NO-OP 而非 SURVIVED |
| selector 命中零条 | `run_test` 以 `returncode == 0` 判通过，而 pytest 在 selector 匹配不到任何测试时退出码为 4 | 拼错的 selector **永久报告 killed**（虚报覆盖）。本轮我新加的 2 个 case 正是这样，先被虚报为 killed | 变异任何文件之前先 `--collect-only` 验每个 selector 至少命中 1 条，否则中止 |

修正后重跑：交接记载的 `18/20` 在原 20 个 case 上**精确复现**（18 killed / 2 survived），所以那个数字本身是对的 —— 缺陷是非确定性的，上一轮恰好没触发。加上本轮 2 个新 case 后为 20/22。

---

## 六、关键设计决策记录

| 决策 | 选择 | 放弃的方案 | 理由 |
|------|------|-----------|------|
| 路由判定 | 三层（静态特征 → 廉价分类器 → 确定性硬门控） | 直接让大模型判断难度 | 模型自评难度不稳定；硬门控保证越界必升级 |
| 升级时的工作区 | 丢弃 shadow 回到干净基线 | 继承 FastPath 的半成品 | 半成品状态会误导 Full Agent；但保留结构化失败诊断 |
| 模型接入 | 自研薄适配层 | LangChain chat models | cache 断点位置与 token 计费细节需要贴着原生 API |
| 验收测试来源 | 外部注入的隐藏测试 | 仓库自带测试套件 | Agent 能读到测试就可能针对性绕过 |
| 项目记忆写入时机 | 仅成功 finalize 或显式教训 | 每轮都写 | 防止记忆膨胀与噪声累积 |
| 子代理权限模型 | `isolation` + `allowed_tools` 双闸，构造期校验 | 运行期检查 | 不合法的规格根本无法构造，早失败 |
| 子代理返回值 | 只回结论 + 成本，不回中间工具流量 | 回完整对话 | 子代理的价值就是隔离上下文开销 |
| 模型档位 | 抽象 tier（fast/balanced/deep） | 预设里写死模型 id | 成本调优集中在一处 |
| MCP 工具命名 | 强制 `mcp__<server>__<tool>` 前缀 | 保留服务器原始名 | 否则第三方 `read_file` 会悄悄劫持内置工具、绕过 `PathGuard` |
| 写类 MCP 服务器 | 必须声明 `shadow_root_arg` 才准入 | 直接信任服务器 | 它持有真实路径，写入会绕过 guard 与 snapshot，让「无非预期修改」失效 |
| MCP 与 FastPath | 完全不可见，注册期即拒绝 | 按需裁剪 schema | 单 server 6–9k tokens 已超 FastPath 全部预算，裁剪也救不回来 |
| skill 的实现形态 | 上下文注入的指令包 | 做成一个 tool | skill 无副作用，做成 tool 会白白多一轮往返 |
| skill 在上下文中的位置 | cache 断点之后 | 放进 system / repo card | 放在断点前会让每个任务的 cache 前缀都不同，缓存彻底失效 |
| skill 与路由的关系 | 同时作为路由证据（`fastpath_safe` / `routing_hint`） | 纯文档 | 命中带具体步骤的 skill 是「证据充分」的强信号，可提升 FastPath 命中率 |
| FastPath 的 skill 加载 | 最多单个 pinned 正文，无索引；命中两个则不 pin | 加载全部匹配项 | 索引约 1.5k tokens 吃不起；两个匹配说明任务没有想象中那么锚定 |
| 工具结果压缩 | 统一入口，MCP 无例外 | 给 MCP 开后门 | 一次返回 50k token 的调用足以单独打爆窗口 |
| FastPath 的补丁形态 | 整文件内容（`files: [{path, content}]`） | unified diff | diff 需要模型精确记住上下文行，失败率高；整文件写入配合 shadow + snapshot 已能约束越界 |
| L2 门控的位置 | 只作用于 L1 输出，不覆盖 L0 | 统一在末尾门控 | L0 的升级判定本身就是确定性的，再过一层是空转 |
| L1 失败时的降级方向 | fallback 到 FastPath | fallback 到 FullAgent | 分类器挂掉不该让每个任务都付全量代价；FastPath 失败仍会升级，最坏情况只多一次廉价尝试 |
| `.env` 加载 | 内联 40 行解析 | 依赖 python-dotenv | 避免为读键值对增加一个运行时依赖；已有 `os.environ` 优先级语义 |
| FastPath 的文件上下文 | 预取 `files_hint` 内联进 prompt（本地读取，零额外调用） | 让模型盲写 / 要求模型先 read_file | 盲写必然导致模型拒绝（"会丢失现有代码"）；read_file 需要工具调用，FastPath 结构化输出模式不支持 |
| FastPath 预取上限 | `MAX_PREFETCH_TOTAL_CHARS=20k` / 单文件 `8k` | 无上限 / 严格 per-file | 超出上限的任务本不适合 FastPath；上限同时防止 prompt 膨胀侵蚀成本优势 |
| spawn_subagent 解耦 | 回调注入（`SpawnCallback`），executor 不导入 agents | executor 直接调用 agents | 防止 tools ↔ agents 循环导入；test 时 mock 更简单 |
| orchestrator 的 FastPath 升级路径 | 丢弃 shadow，建新 WorktreeWorkspace，传递 `escalation_context` | 复用已改动的 shadow | 半成品状态会以不可预测方式误导 Full Agent；干净基线 + 结构化诊断是正确的信息传递 |
| CLI 命令结构 | 单命令（typer 默认），`qqcode --task ...` | 子命令 `qqcode run --task ...` | M6 只有一个用户可见命令；eval/replay 未到达 M5/M6，强行添加子命令只增加文档与测试负担 |
| 定位器结果的去向 | 只进 `prefetch_hint` | 回写 `files_hint` 以便条件 3 生效 | `files_hint` 兼任条件 3 契约。喂进去等于把「浪费 token」换成「拒绝正确补丁」，方向反了 |
| 定位器的排序机制 | IDF + 文件名词干前缀 + 测试文件降权 | 定义点加权 / 标识符精确匹配 | 前者实测 @3 无改进、@5 更差；后者把 @3 从 60% 打到 20%。均已实测否决，非「为简单而省略」 |
| 定位器的停用词表 | 只含功能词与版控动词 | 含缺陷词（`regression`/`missing`/…）的手写表 | 手写表 @3 得 60%，但留一法证明增益来自被测语句自己的词——拟合答案。换表后 60% 由测试降权重现，机制可解释 |
| 树超过 2000 文件时 | 返回空，不排名 | 排被截断的树 / 只排前 N | 截断排名会把 `os.walk` 恰好先到的子树当成整个仓库呈现给模型，是有害的假证据 |
| **定位器摆在哪一层** | **路由层当信号**：答案集中→FastPath 带文件，发散或空→FullAgent | **FastPath 内当能力补丁**（`0f50c4b` 的做法，已判定摆错） | 用户 2026-08-07 指出：产品同时有 FastPath 和 FullAgent，不要求 FastPath 完成所有任务，只要路由合理即可。在 FastPath 内塞定位器 = 让它拿一次性词频猜测去抢 FullAgent（真读文件、能 grep、能迭代）更擅长的活 |
| 路由依据 | 「改动是否小而局部」 | 「任务文本有没有写文件名」 | 后者判断的是提需求的人说话顺不顺手，不是任务难度。5 条 derivable 语句全不含路径，却全是几行的局部修复——正是 FastPath 该吃的活，按「无路径即转 FullAgent」会白让出去 |

---

## 七、风险与未决问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| 路由阈值 τ/L/K 无历史数据可校准 | 冷启动误判率高 | M5 建轨迹库；初期阈值保守。当前值：长度 500 / 文件数 3 / 置信度 0.7 |
| git worktree 在非 git 仓库回退到全量拷贝 | 大仓库启动慢 | 已实现回退；大仓库场景待 M5 实测 |
| ~~**R6：M2 适配层与 `BilledClient` 零测试覆盖**~~ | ~~计费保真度、重试记账、结构化输出互斥均未被测试锁定~~ | ✅ **R6 已关闭**：`tests/test_adapters.py`（43 条）+ `tests/test_billing.py`（24 条）补全，duck-typed fake SDK |
| ~~**R7：FastPath 的 `run_command(["sh", "-c", ...])` 会被 `CommandGuard` 拒绝**~~ | ~~隐藏验收测试路径当前不可用~~ | ✅ **R7 已关闭**：`AcceptanceHarness` 绕过 `CommandGuard`（设计意图：验收命令来自任务作者，信任级别等同于运行 QQCode 的人），注入→执行→清理，agent 全程不可见 |
| ~~**R8：`--mode fast` 无 `files_hint`，修改已存在文件仍会盲写**~~ | ~~fast 模式下编辑任务将产生无效补丁~~ | ✅ **R8 已关闭**（2026-08-06，v1.0.0）：采用待定方案②的变体——`resolve_prefetch_paths` 从任务文本提取文件名并**对真实工作区校验**。范围比 R8 原描述更广：`--mode fast`、L0 skill hint、fallback 三条路径都是无 hint 入口，修复放在 `fastpath.py` 故一并覆盖。实测省 86.2% tokens |
| **R9：15 个 benchmark fixture 里仅 5 个可测量能力** | `behavioral_rate` 混入了「正确修复也会失败」的 fixture，历史 0/15 不是能力数据 | 部分缓解（2026-08-06）：离线审计全部 15 个，判定写入 `benchmarks/tasks/derivability.json`（**不改共享 pin**）；报告改为以 `behavioral_rate_measurable` 领先并列出每个排除项。剩余：3 个 `unverified` 需联网审完；3 个整文件粒度需改 `acceptance_command`（属跨项目决策） |
| ~~**R10：`AcceptanceHarness` 绕过 `CommandGuard` 的安全声明仍未对用户说明**~~ | ~~用户从不可信来源接受验收套件 = 任意代码执行。README 只说明 `CommandGuard` 存在，未说验收通道绕过它~~ | ✅ **R10 已关闭**（2026-08-07）：非空套件首次 `run()` 时向 stderr 打印一次信任级别警告（`TRUST_WARNING`），README 新增「验收套件是可执行代码」小节。原计划的「CLI `--harness` 时警告」不可行——**CLI 没有 `--harness` 参数**，harness 只能经 `run_task(harness=...)` 程序化传入，警告挂在 CLI 会覆盖零条调用路径；挂在 `harness.run()` 覆盖全部入口 |
| 子代理可能被滥用导致成本爆炸 | 单任务成本失控 | `max_turns` 硬上限 + 父级 spawn 数量预算（M4.5 或 M5 补充） |
| 多数写类 MCP 服务器可能不支持根路径参数 | 可用的写类 server 很少 | 接受这个代价——保住三条件收敛优先于多支持一个 server |
| skill 的实际收益未量化 | 可能是纯成本 | **仍未量化**（2026-08-06 核对）：`ReplayEngine.skill_impact()` 已实现，但 `.qqcode/trace.db` 为空（0 行），且该库 gitignored、历史 trace 已随临时仓库丢失。需先跑一批真实任务积累 trace |
| ~~AcceptanceHarness 的安全声明未在 CLI / README 中对用户说明~~ | ~~用户可能从不可信来源接受验收套件~~ | ✅ 与 R10 同一件事（本行是 R10 编号前的旧记法），2026-08-07 一并关闭 |
| **R11：benchmark 从不传 `harness=`，条件 1 在评测里恒空** | 所有 `behavioral_rate` 都不是三条件收敛的产物，而是跑完之后另打 `test_patch` 评出来的；**零逃逸这条指标在这批数据里完全没测**（17 次运行触发信任警告 0 次，反证 harness 通道从未走过） | 已核实：`benchmarks/qqcode_benchmark.py:698` 的 `run_task(...)` 参数表无 `harness=`；全文件仅 `_harness_ran`（223 行定义，795 行调用）在**检测输出里的信任警告**，不是注入。**修法不是简单补上**——把上游 `test_patch` 当 harness 会让 `build_escalation_context` 把 pytest 断言差异喂进 FullAgent 的 prompt，等于告诉 agent 答案，通过率会虚高。需要先设计一个不泄漏断言内容的诊断通道 |
| **R12：未读文件可能被整文件写入静默覆盖** | 模型凭记忆重建一个它没读过的文件 = 销毁真文件。FastPath 写整文件，所以这是数据丢失级缺陷 | 部分缓解（2026-08-07）：prompt 侧已修（`ebf04fd`，删掉「缺席即不存在」那句假话）+ `_prefetch_files` docstring 纠正同一处误推。机械守卫**未落地**，用 `strict=True` xfail 钉住（`62764fa`）。当初判定「守卫不可分离」的依据是「碰撞的 ~10 个测试全部带空 prefetch 到达检查点」。**已验证（2026-08-07，零 API）：前提仍成立，且碰撞面变宽而非收窄 —— 临时插回最小守卫跑全套，660 passed/0 failed → 648 passed/13 failed（定位器落地前是 10）。测试一行未改。** 定位器没有消解碰撞：它填的 3 个预取槽未必是补丁要写的文件。附带发现：失败名单里含 `test_real_hint_still_enforces_condition_three`（`files_hint` 非空的用例），说明「存在于磁盘且不在预取里就拒」这条太粗——真正的守卫至少要豁免 `files_hint` 显式声明的路径（任务已授权）。xfail 维持 |
| **R13：定位器摆错层（`0f50c4b`）** | 它现在在 FastPath 内当能力补丁，而非在路由层当信号 | **降级为「不做」（2026-08-08 实测，零 API）**。原计划是路由层判「答案集中→FastPath 带文件 / 发散或空→FullAgent」。两处实测否决了它：**(1) 集中度阈值不可信** —— `gap_1_2`（top1/top2 分数比）确实分开命中 [1.274, 1.42, 1.431] 与未命中 [1.052, 1.139]，但共测 5 个统计量，3 命中/2 未命中下单个统计量偶然分开的概率 0.2，期望偶然分开数恰为 1.0，实得 1 个；留出集拿不到（bare cache 为 shallow，仅 15 个 commit 且正是 fixture 自身，本仓库 25 个 commit）。**(2) 逐 fixture 推演下它零判决改变** —— skipif 命中 rank 1 会带文件走 FastPath，但那正是 `fastpath.py:386` 今天已在做的事；saferepr 已有 L1 hint 不变；dynamic-xfail miss 会转 FullAgent，而配对数据显示同 fixture 升级 2/2 @39k、直送 2/2 @40k，只省 token；click 的 L1 判了 fullagent，`orchestrator.py:161` 根本不进 `_try_fastpath`，门不执行。按反漂移判据答案是「不能」。搁置的「预取注定为空就转 FullAgent」门同此归类：纯省 token 的配套 |
| **R14：FullAgent 在这批任务上只有一次残缺门下的观测** | 「转给 FullAgent 是正确路由」目前是假设不是结论 | 已有数字但有已知缺陷：full 4/9 = 44.4%、43,080 tokens/run（同 R11 的空条件 1）。同一批里 automatic 是 6/8 = 75%、30,625 tokens——**双档在成功率和成本上同时赢过纯 FullAgent**。**但 2026-08-08 逐 fixture 配对纠正了这个差距的归因**（读 `runs.json` 17 行 + 17 个 `trace.db`）：dynamic-xfail 2/2 vs 2/2、skipif 2/2 vs 2/2、click 0/2 vs 0/2 —— **「升级带诊断」的增量是 0**；全部差距来自 saferepr（automatic 2/2 @5.8k vs full 0/2 @43k，FastPath 直接成功）加上只在 full 臂跑过的 importlib（0/1 @86k，去掉它 full 为 4/8 = 50%）。所以支持双档的证据比原记载更强也更窄：**赢在 FastPath 成功时便宜 7 倍且修好了 FullAgent 做不对的 fixture，不在于失败调用的诊断价值**。样本 n 小，需要 `--cycles ≥ 3` 复测 |
| **R15：`success \| right file` 仅 n=2** | 定位器 60% recall 的全部价值押在这一环上；文件给对了但补丁还是错，前面省的都白算 | 未测（需真实 API）。**但 2026-08-08 读 17 个 `trace.db` 后因果链已闭合到机制层**：全 17 行中 `files_hint_count > 0` 只出现在唯一成功的两次（saferepr，l1_l2 路由 0.86/0.88，5.8k tokens），`= 0` 的四次全部 declined（23k–44k）。`>0 → 2/2` 对 `=0 → 4/4`，n=6 但完全分离，且这是机制不是拟合阈值：模型看见代码就出补丁，看不见就走文档化出口。**这批数据全部早于定位器**（ABBA 于 2026-08-07 15:09，`0f50c4b` 于 2026-08-08 21:16），而定位器对 skipif 命中 rank 1、形状与 saferepr 靠 L1 拿到文件相同。**可检验预测**：skipif 从 24k 的 decline 变为约 6k 的 FastPath 成功。接线已由 `76d4fa5` 用 prompt 级断言 + 变异钉住，所以这个预测测得到东西。这是当前主线的真卡点 |

---

## 八、变更日志

| 日期 | 变更 |
|------|------|
| 2026-08-04 | 需求确认，四项技术选型锁定 |
| 2026-08-04 | M1 完成：协议、工作区、安全管控 |
| 2026-08-04 | 范围新增：子代理支持与预设工作模式 → 规格层并入 M1，执行层排入 M4 |
| 2026-08-04 | 范围新增：本台账，持续维护 |
| 2026-08-04 | M1.5 完成：`ToolRegistry`、artifact 压缩、`MCPServerConfig`、`SkillIndex`、子代理 MCP/skill 授权字段。222 tests 全绿 |
| 2026-08-04 | 路线新增 M4.5（MCP 客户端）；确定 skills 优先级高于 MCP |
| 2026-08-04 | `CostLedger` 新增 `subagent` 阶段；`protocol.py` 新增 `Tier`/`ALL_TIERS` |
| 2026-08-04 | M2 代码完成：Anthropic / OpenAI 双适配器、`BilledClient`、usage 归一化 |
| 2026-08-04 | **R6 关闭**：补写 `tests/test_adapters.py`（43 条）+ `tests/test_billing.py`（24 条），duck-typed fake SDK；M2 不变量从「代码审读」升级为「测试锁定」 |
| 2026-08-04 | M3 完成：L0/L1/L2 三层路由、FastPath 三条件门控（含 `acceptance_tampering` 防篡改）、`.env` 配置层。319 tests |
| 2026-08-04 | **R7 关闭**：`AcceptanceHarness` 建立隐藏验收执行通道（绕过 `CommandGuard` 属设计意图）；注入→执行→cleanup，agent 不可见 |
| 2026-08-04 | FastPath 三条件门控修正：① 条件 2（agent 完成状态）之前从未检查；② `changed_files` 类型谎言（`set` → `frozenset`）；③ acceptance 目录篡改防护；④ 验收后重新快照；⑤ 无 `files_hint` 时条件 3 标注为 unenforceable 而非隐式通过 |
| 2026-08-04 | M4 完成：10 个内置工具 + `ToolExecutor`（guard 执行 + artifact 压缩 + spawn 回调）、`execute_full_agent`（ReAct + 5 个终止条件）。350 tests |
| 2026-08-04 | M6 完成：`orchestrator.run_task`（三模式 + 升级路径）+ typer CLI + rich 输出。真实 API 端到端通过（auto/fast/full 三模式 · finalize 原子写回验证） |
| 2026-08-04 | **FastPath 根因修复**：发现 FastPath 对已存在文件的修改必然 declined（无文件上下文导致模型合理拒绝）；增加 `_prefetch_files` 在 prompt 组装时内联文件内容（零额外调用），修后 FastPath 对 docstring 类任务一次命中 |
| 2026-08-04 | **R8 新增**：`--mode fast` 无 `files_hint`，修改已存在文件仍会盲写；fast 模式当前适合新建文件任务，待定修法 |
| 2026-08-04 | `filter_acceptance_paths` 类型签名修正：接受 `frozenset[str] \| set[str]`，返回 `frozenset[str]`，消除 orchestrator.py 中的 mypy 类型错误 |
| 2026-08-04 | 台账同步：M2–M6 交付物、关键修正、R6/R7 关闭记录全部落档 |
| 2026-08-05 | FastPath 路由实验落档 `docs/EXPERIMENT_FASTPATH_ROUTING.md` |
| 2026-08-05 | **三个 OpenAI 专属缺陷修复**：① 工具结果标 `Role.USER` 被 OpenAI 适配器静默丢弃；② `OutputSpec` schema 缺 `additionalProperties: false`，OpenAI `strict` 模式 400；③ FastPath 要求 `stop_reason == "tool_use"`（Anthropic 约定），拒掉所有合法 OpenAI 补丁。三者共同根因：共享代码按 Anthropic wire 约定编写 |
| 2026-08-06 | M7 会话交互层完成：REPL、sessions、`/undo`、实时工具输出；shadow 改为从工作树 seed（修好一个既有的未提交改动丢失缺陷） |
| 2026-08-06 | `DEFAULT_MODEL` / `~/.config/qqcode/env`：一份全局配置即可指向非厂商端点 |
| 2026-08-06 | **R8 关闭 + v1.0.0 发布**：`resolve_prefetch_paths` 从任务文本提取文件名并对工作区校验，覆盖 `--mode fast` / L0 skill hint / fallback 三条无 hint 路径。真实 API A/B：automatic 从 3/3 declined→升级 变为 3/3 FastPath 命中，**−16,288 tokens（−86.2%）**。关键设计：解析结果**不回写** `files_hint`——该字段兼任条件 3 的强制契约，填入猜测会把「declined」换成「错误拒绝」 |
| 2026-08-06 | **R9 新增并部分缓解**：离线审计 15 个 fixture 的隐藏断言可推导性（5 可推导 / 4 不可推导 / 3 整文件粒度 / 3 未审）。判定存入 `benchmarks/tasks/derivability.json`；报告改以 `behavioral_rate_measurable` 领先。**同时修评测器**：`_run_acceptance` 原先把「隐藏 test_patch 打不上」折叠成 `passed=False`，即把仪器故障记成能力不足，现归为 `incident_type="test_conflict"` |
| 2026-08-06 | **发现 `benchmarks/tasks/real_tasks_v2.json` 是指向 `claude-engineer` 项目的符号链接**（git 模式 `120000`）。写它会静默修改另一个项目，且本仓库 `git diff` 完全干净——常规的「提交前看 diff」无法发现。故 fixture 审计结论另存独立文件，共享 pin 一字未改 |
| 2026-08-06 | **R10 新增**：`AcceptanceHarness` 绕过 `CommandGuard` 的信任级别至今未对用户说明（M6 曾要求「显著标注」）。属真实安全缺口，待修 |
| 2026-08-07 | **R10 关闭**：`AcceptanceHarness.run()` 在非空套件首次执行时向 stderr 打印一次 `TRUST_WARNING`；README「安全隔离」新增小节，说明验收命令不走 `CommandGuard`、为何刻意如此（走 `run_command` 会让测试出现在 agent 的工具日志里，agent 便能迎合测试）、以及仍生效的两条限制（剥离密钥、超时）。**修法与原计划不同**：CLI 并无 `--harness` 参数，警告只能挂在 harness 自身才覆盖真实调用路径。警告走 stderr 而非 stdout，避免污染机读输出；进程内只打一次，批量跑不会刷屏。3 条新测试均做过变异验证（去掉警告调用 / 去掉一次性守卫 / 去掉空套件守卫，各自对应的测试都失败）。604 tests 绿，mypy 0 |
