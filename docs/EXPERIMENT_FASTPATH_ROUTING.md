# FastPath 路由实验报告

**日期**：2026-08-05
**数据**：`benchmarks/results/trivial-paths-20260805-190718/runs.json`
**规模**：4 模型 × 24 任务 × 2 模式 = 192 次运行，5 次基础设施故障（已排除）

---

## 1. 实验目的

量化 QQCode 两层架构的收益，并用多模型互证结论不依赖单一模型：

1. FastPath（单次调用生成完整补丁）能否独立完成任务
2. Full Agent（多轮工具循环）能否独立完成任务
3. FastPath 主动放弃时，能否逃逸到 Full Agent 兜底
4. 相比全程 Full Agent，智能路由节省多少 agent loop、模型调用与 token

## 2. 实验设计

### 2.1 两种模式对比

| 模式 | 行为 | 作用 |
|---|---|---|
| `full` | 强制 Full Agent | 基线：不做路由时任务集的总成本 |
| `auto` | 路由 + FastPath + 逃逸 | 真实用户路径 |

**刻意排除的第三种模式**：`fast`（空 `files_hint`）。它看起来最省，但只是因为提示更短、未触发模型的 extended thinking —— 那是真实用户走不到的路径上的测量假象，不是节省。`full` vs `auto` 两侧都是真实代码路径，思考 token 是各自路径的真实成本。

### 2.2 三个计数器

| 指标 | 含义 | 可比性 |
|---|---|---|
| `model_calls` | 所有 provider 调用（路由 + FastPath + Full Agent 每轮） | **跨模型可比**，不受网关开销影响 |
| `turns_used` | Full Agent 工具循环轮数 | FastPath 单次调用无循环，恒为 0 |
| `tokens_total` | `automatic_total = input + output` | **跨 provider 不可比**，见 §5.1 |

FastPath 成功一次 = 2 次调用（1 次 L1 路由分类 + 1 次生成补丁）、0 轮循环。

### 2.3 任务集：24 个自建任务

不使用 SWE-bench 真实 issue，原因见 §5.3。任务分三档：

- **简单（8）**：单文件单函数，如 `a - b` → `a + b`、闭区间边界、去重保序
- **中等（8）**：多文件、跨文件一致性、有状态逻辑，如跨 3 文件重命名、新建 `config.py` 抽公共常量、类属性改实例属性、递归展开、上下文管理器
- **惯用法陷阱（8）**：表面合理的修法会被隐藏测试判死
  - `mutable-default`：可变默认参数跨调用共享
  - `none-vs-falsy`：`if not value` 误判 `0` / `''` / `[]`
  - `truncate-suffix`：`'...'` 须计入长度上限
  - `retry-with-backoff`：最后一次尝试后不得 sleep，且须重抛异常
  - `cross-module-flag`：须新建文件；测试运行时翻转 flag，`from x import y` 会失败
  - `validate-then-store`：校验失败必须无副作用
  - `case-insensitive-dedupe`：保留首次出现的大小写与顺序
  - `counter-reset-isolation`：类属性 dict 被所有实例共享

### 2.4 fixture 有效性校验（关键前置）

每个隐藏测试必须**在原始代码上失败、在参考修复上通过**，且不留残留文件。24/24 全部通过。

这一步不可省略。SWE-bench 那批 fixture 的失效模式正是：隐藏测试断言了任务说明从未提及的实现细节，导致正确修复被判为失败。

### 2.5 系统提示词约束

依据实际观察到的失败模式补充，非凭空设计：

**Full Agent** — Scope 段：只改任务要求的部分，禁止顺手重构；**禁止增删改测试文件**（针对上一轮 agent 自写测试导致隐藏 test_patch 无法应用）。Efficiency 段：不重读已读文件、同一错误失败两次即换方法、验证完立即 `finish`（针对 7 次 `max_turns` 与 5 次 `stuck`）。

**FastPath** — 整文件覆写时须保留未要求改动的内容（imports、docstring、无关函数）；显式处理任务点名的边界情况。

> 效果无法归因：这是提示词改动后的首次运行，且 8 个任务同期新增，无同条件对照。**不声称提示词带来了改进。**

---

## 3. 主结论：FastPath 在所有模型上均有效

### 3.1 模型调用与 agent loop（跨模型可比）

仅统计两种模式都行为通过的任务。

| 模型 | 任务 | full calls | auto calls | 节省 | full loops | auto loops | 节省 |
|---|---|---|---|---|---|---|---|
| gpt-5.6-sol | 24 | 154 | 111 | **27.9%** | 154 | 66 | **57.1%** |
| gpt-5.6-terra | 24 | 142 | 106 | **25.4%** | 142 | 61 | **57.0%** |
| gpt-5.6-luna | 20 | 123 | 75 | **39.0%** | 123 | 37 | **69.9%** |
| claude-sonnet-5 | 20 | 129 | 71 | **45.0%** | 129 | 33 | **74.4%** |

**四模型、两 provider、两种 API 协议，全部为正，无一例外。**

### 3.2 收敛性即互证

只看 FastPath 一次通过的任务，调用次数节省高度一致：

| 模型 | 节省 |
|---|---|
| gpt-5.6-sol | 68.5% |
| gpt-5.6-terra | 65.0% |
| gpt-5.6-luna | 67.4% |
| claude-sonnet-5 | 68.5% |

四个独立模型落在 **3.5 个百分点**内。这反映的是架构常量 —— Full Agent 需 5–8 次调用，FastPath 恒定 2 次 —— 与模型能力无关。这一收敛本身就是最强的互证。

### 3.3 Token 节省（FastPath 成功任务）

| 模型 | full tok | auto tok | 节省 |
|---|---|---|---|
| gpt-5.6-luna | 99,551 | 13,915 | **86.0%** |
| gpt-5.6-terra | 166,893 | 43,725 | **73.8%** |
| gpt-5.6-sol | 170,303 | 53,869 | **68.4%** |
| claude-sonnet-5 | 97,398 | 113,859 | −16.9%（测量失真，见 §5.1） |

`gpt-5.6-luna` 是唯一无网关开销的路由，其 **86.0%** 最接近真实架构节省。

### 3.4 逃逸兜底：四模型全部验证

| 模型 | FastPath 通过 | 逃逸 | 逃逸率 |
|---|---|---|---|
| claude-sonnet-5 | 18/24 | 6 | 25% |
| gpt-5.6-luna | 15/21 | 6 | 29% |
| gpt-5.6-sol | 14/24 | 10 | 42% |
| gpt-5.6-terra | 14/24 | 10 | 42% |

**所有逃逸最终都完成了任务。** 触发原因是 FastPath 正常判断「单次不够」（多为需新建文件或多处协同修改），而非故障。

代价：逃逸任务比直接走 full 贵 11k–53k token，因需支付两段费用。但整体仍净省 —— 押对次数远多于押错。

### 3.5 行为正确率

| 模型 | 正确率 |
|---|---|
| gpt-5.6-sol | 48/48 |
| gpt-5.6-terra | 48/48 |
| claude-sonnet-5 | 44/48 |
| gpt-5.6-luna | 42/43（5 次 provider incident 已排除） |

8 个惯用法陷阱任务全部通过，说明 agent 并非表面模式匹配。

---

## 4. 一句话结论

不同模型、不同 provider 下 FastPath 均有效：**模型调用次数节省 25–45%**（隔离到 FastPath 成功任务时收敛至 65–68.5%），**agent loop 节省 57–74%**，**token 在三个无污染路由上节省 68–86%**，逃逸兜底四模型全部验证成功。

---

## 5. 有效性限制（必读）

### 5.1 跨 provider 的 token 数字不可比

裸 SDK 测试：仅发送 `"hi"`（2 token）

| 路由 | 真实开销 | 计入 `automatic_total` |
|---|---|---|
| sol / terra | 4,387（其中 3,840 cached） | 547 |
| **luna** | **7** | **7** ← 唯一干净 |
| claude-sonnet-5 | 2,540（其中 2,538 cache_read） | 2 |

各网关注入隐藏前缀，而 `automatic_total = input + output` 不含 cache token，导致：

- **Claude 被系统性低估**：每次调用约 2,538 token 记为 `cache_read` 而漏出总数。`auto` 调用次数少（2 次 vs full 的 5–8 次），漏计也少，反而显得更贵 —— 这就是 −16.9% 的来源，**不是 FastPath 无效**。同模型的 `model_calls` 显示其节省 68.5%，与其他三个模型一致。
- **sol / terra 被高估**：网关虚报 `prompt_tokens`。

**同一模型内的 full vs auto 对比有效**（两侧承受同样开销）；跨 provider 的 token 绝对值不可比。

### 5.2 extended thinking 自主触发造成方差

同一代码路径、同等输入规模，token 可差 3–4 倍。例：`full` 模式多数任务 5,900–10,300，个别达 20,649–25,266。模型自主决定是否花费思考 token，且该决策为黑盒、不可控。逃逸路径叠加此波动时单次成本可翻 4 倍。

### 5.3 任务集偏简单，结论不可外推至真实工程任务

24 个任务均为自建合成任务，文件内容已内联在提示中、改动范围明确。FastPath 一次通过率 58–75%，说明任务对其偏易。

对比参照：同一套代码在 SWE-bench 真实 pytest/flask issue 上，24 次运行仅 3 次通过。那类任务需在陌生大仓库中自行定位代码，与此处不在同一量级。

**准确表述**：在明确定义、小范围的任务上，路由可省 25–45% 调用与 57–74% loop。真实仓库任务的节省幅度尚无可信数据。

### 5.4 提示词效果未做 A/B

见 §2.5。

---

## 6. 附带修复的产品缺陷

实验过程中定位到 3 个只在 OpenAI 路径暴露的缺陷，根因同一：共享代码按 Anthropic 的 wire 约定编写，Anthropic 容错而 OpenAI 100% 失败。

| # | 位置 | 缺陷 |
|---|---|---|
| 1 | `qqcode/agents/graph.py:160` | 工具结果标记为 `Role.USER`。Anthropic 将 `USER` 与 `TOOL` 折叠为同一 wire role；OpenAI adapter 仅对 `Role.TOOL` 生成 `{"role":"tool","tool_call_id":...}`，其余角色的 `ToolResultContent` 被**静默丢弃** —— 我们实际发出空消息，网关回 `No tool output found for function call fc_...` 是正确响应 |
| 2 | `qqcode/models/openai_adapter.py` `to_response_format()` | `strict: true` 要求每个 object 显式带 `additionalProperties: false`；Anthropic 的 `input_schema` 无此要求。同时打死 FastPath 与 L1 路由 |
| 3 | `qqcode/routing/fastpath.py` | 硬编码要求 `stop_reason == "tool_use"`。Anthropic 的结构化输出是强制工具调用故成立；OpenAI 返回普通内容（`stop` → `end_turn`），导致有效补丁被误判为截断 |

缺陷 2 连带打死路由层：`router.py` 的 `except Exception: return None` 将 400 一并吞掉，12/12 运行的 `tokens_routing=0` 证明 L1 分类器一次都未成功。

修复原则：provider 特有要求置于 adapter 层（`to_response_format` 深拷贝后递归注入 `additionalProperties`，不修改共享 schema 对象），并将静默丢弃改为显式抛错。

配套可观测性修复：

- `finish_summary` 贯通至 trace DB（含 migration），失败原因不再丢失
- benchmark 补上被丢弃的 `RunResult.error`
- `acceptance_output` 由 `out[:500]` 改为 `_clip(out, 1200)` 保留首尾 —— pytest 判决行在末尾
- 新增 provider 故障分类，网关故障不计入 behavioral rate
- 新增 `_InjectionSniffer`：某网关会将拒绝话术作为**正常 assistant 内容**返回（无 tool call），agent 循环将其视为空回合并判 `stuck`，与真实失败无法区分。检测置于 wire 边界，因 `RunResult` 不携带 transcript

---

## 7. 复现方式

```bash
# 单模型
python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-luna --effort high

# 四模型矩阵（约 90 分钟）
python benchmarks/trivial_paths_benchmark.py \
  --models gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna,claude-sonnet-5 --effort high

# 单任务冒烟
python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-luna --tasks fix-subtract
```

`runs.json` 每次运行后落盘，中断不丢已完成数据。

## 8. 后续建议

1. **优先用 `gpt-5.6-luna` 做成本实验** —— 唯一无网关开销的路由，token 数字可直接采信（代价：5/48 的 incident 率）
2. 若需跨 provider token 对比，需先决定是否将 cache token 计入 `automatic_total`（注意：cache 读取实际计费低于常规 input，全额计入会高估成本）
3. 修复 SWE-bench fixture 规范质量问题后，补测真实仓库任务的节省幅度
4. 对系统提示词做 A/B 以量化其贡献
