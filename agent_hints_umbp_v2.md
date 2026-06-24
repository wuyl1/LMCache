# UMBP Agent Hints 方案 v2：Scheduler 驱动的语义缓存策略

## 1. 背景与目标

v1 方案把核心组件设计成 **Semantic KV Adapter**：adapter 位于推理框架和 UMBP 之间，负责把 Agent 语义翻译成 `semantic spans`、`canonical key`、`priority / TTL` 和 `skip / report / put` 决策，再通过 UMBP API 与 UMBP 交互。

v2 方案重新划分控制面：**不再把 adapter 作为 UMBP 的主要交互层，而是让 scheduler 成为 agent hints 的消费方和策略决策方**。这更接近 NVIDIA Dynamo 的思路：上层通过请求级 hints 暴露 session、priority、输出长度和生命周期等信息，serving 层 scheduler/router 根据这些 hints 做路由、排队和 KV cache 策略优化。

同时，v2 不把 UMBP 限定为当前已有接口的集合。它保留现有 external metadata path 和 pointer-based KV data path，同时提出面向 agent hints 的 policy control 扩展，让 scheduler 可以显式保护、降级、释放或调整 KV 的生命周期。

v2 的目标是：

1. **贴合现有软件架构**：系统已有 scheduler，缓存策略应由 scheduler 统一决策，而不是新增 adapter 分裂控制面。
2. **分层接收 hints**：先支持稳定、容易生产的 base hints，再扩展到包含语义和 key 管理的 advanced hints。
3. **协同扩展 UMBP 控制面**：UMBP 继续承担执行层职责，但可以新增 `pin / demote / evict / update phase TTL` 等 key/block-level policy controls。
4. **围绕 semantic-driven cache policy optimization**：语义用于决定缓存价值、生命周期、准入和 eviction/pin 策略；复用安全仍由 token chunk、model、tokenizer、KV layout 和隔离字段保证。

---

## 2. 总体架构

v2 的核心链路是：

```text
+------------------------------------------------------------+
| Agent / Harness / Agent Orchestrator                       |
|                                                            |
| base hints: session_id / session_timeout / priority        |
| advanced:   semantic spans / reuse_scope / key fields      |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
| Request Frontend                                           |
|                                                            |
| attach hints with request, tokens, and KV layout metadata  |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
| Scheduler                                                  |
|                                                            |
| decide route / prefetch / report / put / revoke / tiering  |
| decide phase TTL / priority / pin / eviction               |
+-----------------------------+------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+----------------------------+       +------------------------------------+
| Inference Workers          |       | UMBP                               |
| SGLang / vLLM / others     |       |                                    |
|                            |       | Existing data/index APIs:          |
| - prefill / decode         |       | - match/report/revoke external KV  |
| - local KV cache           |       | - RoutePut / RouteGet              |
| - expose KV layout / ptr   |       | - batch_put/get ptr APIs           |
| - consume fetched KV       |       | - peer-owned tier storage          |
|                            |       |                                    |
|                            |       | Proposed agent-hint controls:      |
|                            |       | - pin / unpin by key or block      |
|                            |       | - demote / promote across tiers    |
|                            |       | - evict / expire / update phase_ttl|
|                            |       | - priority and admission policy    |
+-------------+--------------+       +-----------------+------------------+
              |                                       ^
              | metadata report: external KV hash,    |
              | tier, node id, block metadata         |
              +-------------------------------------->|
              |                                       |
              | put/get data path: canonical key,     |
              | KV layout, src/dst pointer            |
              +-------------------------------------->|
              |                                       |
              | fetched KV bytes copied into worker   |
              | buffer; match result guides routing   |
              |<--------------------------------------+
              |                                       |
              | local cache pressure / produced KV    |
              | and block lifecycle events            |
              +-------------------------------------->|
                                                      |
Scheduler -> UMBP policy API:
  pin / demote / evict / update phase TTL / update priority

UMBP -> Scheduler feedback:
  policy result / matched candidates / tier state / pressure signal
```

这个架构中，scheduler 是策略中枢，worker 和 UMBP 是并行执行子系统：

- 它像 Dynamo Router 一样消费请求级 hints，做 routing、queueing、KV locality 和 session affinity。
- 它也消费 advanced semantic hints，做更细粒度的 KV admission、phase TTL、priority、pin 和 eviction/tier 策略。
- Request Frontend 对应 Dynamo 中的 `dynamo.frontend` 职责：接收 OpenAI/Anthropic-compatible 请求，解析 hints 和 session metadata，并把请求交给 scheduler/router。
- 推理 worker 负责 prefill/decode、本地 KV cache、KV layout / pointer 暴露，以及消费从 UMBP 取回的 KV。
- UMBP 的 master 更像 routing advisor，维护 external KV index 和 owned-key routing state；真正的 KV slot、page location、tier storage 和 eviction 由 peer 持有和执行。
- v2 不局限于当前 UMBP 已有 API，而是把 UMBP 作为可协同演进的缓存子系统：保留现有 pointer-based data path 和 external metadata path，同时补齐 agent-hint-driven policy controls。
- UMBP 不直接解析 Agent 语义，只执行 scheduler 下发的 opaque key、metadata 和缓存策略动作。

---

## 3. Hint 分层

v2 把 hints 分为两层：

```text
AgentHints
  ├── base      # 稳定、请求级、scheduler 可直接消费
  └── advanced  # 语义增强、KV identity、细粒度缓存策略
```

### 3.1 Base Hints

Base hints 参考 Dynamo 的 `nvext.agent_hints` 和 `nvext.session_control` 思路，目标是先解决 routing、queueing、session affinity 和粗粒度生命周期。

建议字段：

```text
base:
  session_id              # 会话标识，用于 sticky routing、session-scoped reuse、生命周期管理
  session_action          # open / bind / close
  session_timeout_ms      # session inactivity timeout
  priority                # 请求或缓存保留优先级
  expected_output_tokens  # 类似 Dynamo osl，用于输出 KV block 负载估计
```

这些字段不要求上层提供完整语义，只要求提供 serving 层容易理解的信息：

- `session_id` 让 scheduler 做 sticky session routing，减少同一会话 KV 在 worker 间漂移。
- `priority` 影响 scheduler queue ordering、UMBP eviction/tier 策略。
- `expected_output_tokens` 帮助 scheduler 估算 decode 阶段的 KV 增长和 worker 未来负载。
- `session_action` / `session_timeout_ms` 支持 session open/close、subagent 生命周期和自动清理。

Base 层的生命周期语义应贴近 Dynamo 的 `session_control.timeout`：`session_timeout_ms` 表示 session 多久不活跃后可以关闭，用于 sticky routing、session KV isolation 和清理。它不是某个 semantic span 或 KV block 的缓存 TTL。

不同执行上下文可以使用不同的 `session_id` 和 `session_timeout_ms`。例如用户请求“分析数据库里的销售异常并生成报告”时，主 Agent 与 SQL subagent 的 KV 生命周期不同，不应强行共用同一个 session。

这个例子里报告任务一共有 **2 条 session 生命周期**，即 2 个不同的 `session_id`：`report-main` 和 `report-sql-subagent`。为了说明 `priority`，时间线中还插入一条无关的低优先级后台 session：`background-summary`。这不表示只有 2 次请求；每一轮 LLM 请求都会携带 session hints，区别在于 lifecycle action 只在开关边界出现：

- `session_id`：每一轮都带，用来找到已有 sticky binding 或 session slot。
- `session_action`：只在生命周期边界带；`bind/open` 表示开始，`close` 表示结束，中间轮次省略表示 continue。
- `session_timeout_ms`：通常只在 `bind/open` 时设置，用作 inactivity timeout 和漏发 `close` 时的兜底清理，不是 KV block TTL。
- `priority`：每一轮都可以带，用于 scheduler queue ordering，也可以影响 KV tier placement 和 eviction 优先级；它不是 session 生命周期动作。
- `expected_output_tokens`：每一轮都可以带，用于 scheduler 估算 decode 阶段的 KV / GPU cache / worker 时间。

这个 base-hints 例子主要覆盖 scheduler 图中的四类决策：

- `decide route`：由 `session_id`、sticky binding 和 worker 负载决定请求去哪个 worker。
- `session lifecycle`：由 `session_action` 和 `session_timeout_ms` 决定 bind/open/close、inactivity cleanup。
- `priority queueing`：由 `priority` 决定排队顺序；数值含义由 serving 系统定义，本例假设数值越大优先级越高。
- `decode load estimate`：由 `expected_output_tokens` 估算未来输出 KV 增长和 worker 负载。

按请求时间线和 session 轮次展开如下：

```text
req1: 用户请求“分析数据库里的销售异常并生成报告”，主 Agent 理解任务并规划步骤
  hints:
    session_id = report-main
    session_action = bind
    session_timeout_ms = long
    priority = 5
    expected_output_tokens = 256
  scheduler decisions:
    - decide route: 根据 KV overlap、当前负载和预计 decode 负载选择 worker-A，并建立 report-main -> worker-A sticky binding
    - session lifecycle: session_action=bind 只做路由亲和，不创建后端 streaming session slot
    - session lifecycle: session_timeout_ms=long 表示 report-main 长时间不活跃后才兜底清理
    - priority queueing: priority=5 表示这是用户可见的交互式报告任务，排队时优先于后台低优先级任务
    - decode load estimate: expected_output_tokens=256 表示这一轮主要做任务规划，decode 负载较小

req2: 主 Agent 委派 SQL subagent 查询销售表，生成并执行销售异常查询
  hints:
    session_id = report-sql-subagent
    session_action = open
    session_timeout_ms = short
    priority = 6
    expected_output_tokens = 512
  scheduler decisions:
    - decide route: 根据 KV overlap、当前负载和预计 decode 负载选择 worker-B，并建立 report-sql-subagent -> worker-B sticky binding
    - session lifecycle: session_action=open 触发后端 session slot，用于 SQL subagent 的 KV 隔离
    - session lifecycle: session_timeout_ms=short 表示 SQL session 短时间不活跃即可兜底清理
    - priority queueing: priority=6 表示 SQL 结果阻塞最终报告，可在队列中略高于主 Agent 的普通整理轮次
    - decode load estimate: expected_output_tokens=512 帮助 scheduler 估算 SQL 生成和结果解释带来的 decode KV 增长
    - 此时 report-main 仍然存在；两条 session 生命周期并存

req3: SQL subagent 把异常销售数据和解释返回给主 Agent，结束 SQL 子任务
  hints:
    session_id = report-sql-subagent
    session_action = close
    session_timeout_ms = omitted
    priority = 6
    expected_output_tokens = 128
  scheduler decisions:
    - decide route: session_id 指向要关闭的已有 report-sql-subagent session
    - session lifecycle: session_action=close 解除 sticky binding，关闭后端 session slot
    - session lifecycle: 释放 SQL subagent 的 session-scoped KV；timeout 不再重要
    - priority queueing: priority=6 让这个阻塞主报告的收尾请求不要被后台任务拖延
    - decode load estimate: expected_output_tokens=128 表示这轮只返回整理后的 SQL 结论，预计输出较短

req4: 另一个后台任务请求生成历史销售日志摘要，用于离线归档，不阻塞当前报告
  hints:
    session_id = background-summary
    session_action = bind
    session_timeout_ms = medium
    priority = 1
    expected_output_tokens = 1024
  scheduler decisions:
    - decide route: 根据 session_id 建立 background-summary 的 sticky binding
    - session lifecycle: session_action=bind 只做路由亲和，不创建后端 session slot
    - priority queueing: priority=1 表示后台低优先级任务；当它与 report-main 或 report-sql-subagent 同时排队时，scheduler 优先处理 priority=5/6 的报告链路
    - decode load estimate: expected_output_tokens=1024 表示后台摘要可能生成较长输出，scheduler 可以把它的未来 decode 负载计入 worker 选择

req5: 主 Agent 接收 SQL 结果，整理异常原因、影响范围和报告结构
  hints:
    session_id = report-main
    session_action = omitted
    session_timeout_ms = omitted
    priority = 5
    expected_output_tokens = 512
  scheduler decisions:
    - decide route: session_id 命中 req1 建立的 report-main binding，继续回到 worker-A
    - session lifecycle: session_action 省略表示继续已有 main session，不重新 bind/open
    - session lifecycle: 沿用 long timeout 语义，并刷新 inactivity 计时窗口
    - priority queueing: priority=5 延续用户可见报告任务的排队优先级
    - decode load estimate: expected_output_tokens=512 表示这一轮主要整理结构，输出长度中等

req6: 主 Agent 生成最终报告；如果任务不再继续，关闭主 Agent session
  hints:
    session_id = report-main
    session_action = close
    session_timeout_ms = omitted
    priority = 5
    expected_output_tokens = 2048
  scheduler decisions:
    - decide route: session_id 指向要关闭的已有 report-main session
    - session lifecycle: session_action=close 结束 report-main 的 sticky binding
    - session lifecycle: 如果后端曾为该 session 管理 session-scoped KV，也在这里释放
    - priority queueing: priority=5 用于最终响应的排队；close 仍由 session_action 控制
    - decode load estimate: expected_output_tokens=2048 表示最终报告输出较长，scheduler 应把较大的 decode KV / GPU cache 占用纳入负载估计

bind  = 只做路由亲和
open  = 路由亲和 + 后端 session KV 隔离
close = 结束已有 session 并释放资源
```

`session_timeout_ms` 主要作为异常退出、漏发 close 或等待下一轮请求时的兜底清理机制。

Base hints 的价值在于低门槛：即使没有 token span 级语义，scheduler 也能先获得 routing locality、session isolation、priority queueing 和生命周期控制收益。

### 3.2 Advanced Hints

Advanced hints 参考 Sutradhara 的语义 phase 和 v1 第七节中的 LMCache-style key 管理，目标是让 scheduler 做 semantic-driven cache policy optimization。

建议字段：

```text
advanced:
  semantic_spans:
    - token_start
      token_end
      phase              # SYSTEM_PROMPT / USER_QUERY / TOOL_OUTPUT / PARTIAL_PREFILL / RESPONSE / UNKNOWN
      source             # role:system / role:user / tool_event / rule / explicit_hint
      cacheable          # optional, 上层是否显式认为可缓存

  key_identity:
    model_id
    tokenizer_id
    kv_layout_version
    token_chunk_hash
    prefix_hash
    reuse_scope          # GLOBAL / TENANT / AGENT / SESSION / REQUEST
    cache_salt           # 可选，用于租户或用户隔离

  policy:
    phase_priority
    phase_ttl_ms
    pin_until_event      # tool_returned / agent_step_done / session_close
    admission            # skip / report / put，可由上层建议，也可由 scheduler 决定
```

Advanced hints 在 base hints 之上提供缓存价值判断和安全复用所需的信息：

- **语义价值识别**：识别 `SYSTEM_PROMPT`、`TOOL_OUTPUT`、`RESPONSE` 等不同 KV 的复用价值。
- **语义生命周期策略**：把 phase 转成 phase TTL、priority、pin 或 demote/evict 策略。
- **语义安全复用**：把 phase、reuse scope、tenant/session 与 LMCache-style token chunk key 结合，避免“语义相似”导致错复用。
- **语义准入控制**：决定 chunk 是 `skip`、`report` 还是 `put`。

也就是说，`semantic_spans` 只描述 token 区间和 phase；`phase_ttl_ms`、`pin_until_event`、`admission` 这些缓存策略由 scheduler 根据 phase、默认策略、命中率和资源压力衍生出来。上层可以显式建议这些 policy，但 scheduler 应保留最终裁剪和覆盖权。

Advanced hints 不要求一次性全部具备。没有显式 semantic spans 时，scheduler 可以根据 message role、tool event 和默认规则生成保守 phase；没有完整 key_identity 时，只允许 metadata-only routing，不进入 UMBP-owned KV bytes 托管。

---

## 4. Scheduler 如何消费 Hints

Scheduler 是 v2 的核心策略层。它不只是排队组件，而是 agent hints 到 UMBP 缓存动作的决策中心。

### 4.1 Routing 与 Session Affinity

Base hints 中的 `session_id` 用于 routing：

```text
if session_id has sticky worker:
    route to sticky worker
else:
    use UMBP external KV match + worker load + prefill/decode cost
    bind session_id to selected worker when needed
```

对应优化：

- 同一 session 后续请求尽量回到已有 KV 的 worker。
- subagent 可以拥有独立 `session_id`，结束时通过 `session_action=close` 清理相关 KV。
- 当 session 过期或关闭时，scheduler 撤销 external KV metadata，释放或降级 session-scoped KV。

这部分接近 Dynamo 的 sticky session 和 KV-aware routing。

### 4.2 Queueing 与 Priority

Base hints 中的 `priority` 用于 scheduler 排队：

```text
queue_order = policy_score(priority, arrival_time, token_cost)
```

可能策略：

- 高优先级请求先调度。
- 长 prefill、低优先级请求可延后或降级。
- 对 latency-sensitive 请求提高 routing 和 cache retrieval 优先级。

`priority` 也可以作为 UMBP policy hint，用来影响 tier placement 和 eviction 顺序：

- 高 priority KV 优先保留在 HBM/DRAM。
- 低 priority KV 更早下沉到 SSD 或被撤销 metadata。

### 4.3 Output Load Estimation

Base hints 中的 `expected_output_tokens` 类似 Dynamo 的 `osl`。Scheduler 可以用它估算 decode 阶段新增 KV block：

```text
estimated_output_blocks = ceil(expected_output_tokens / block_size)
```

用途：

- 避免把大量长输出请求集中到同一个 worker。
- 在 routing 时同时考虑 prefill KV overlap 和 decode 未来负载。
- 在 UMBP tier 策略中预留或限制热层空间。

### 4.4 Lifecycle 与 Phase TTL

Base `session_timeout_ms` 和 advanced `phase_ttl_ms` 是两层不同的生命周期 hint：

- `session_timeout_ms` 对应 Dynamo `session_control.timeout`，表示 session 多久不活跃后可以关闭。
- `phase_ttl_ms` 是 semantic span 级缓存 TTL，由 `semantic_spans.phase` 衍生，用来决定某段 KV 多久还值得保留。

因此，`session_timeout_ms` 应按执行上下文设置：主 Agent session 可以较长，短生命周期 subagent session 可以较短，并在 subagent 完成时显式 `close`。`phase_ttl_ms` 则按 `SYSTEM_PROMPT`、`USER_QUERY`、`TOOL_OUTPUT` 等 phase 生成，作用于具体 KV span。

Scheduler 可以把 phase TTL 转成不同所有权模式下的动作：

- metadata-only 模式下，phase TTL 到期后 revoke external KV metadata。
- UMBP-owned 模式下，phase TTL 到期后降低 priority、迁移到冷 tier，或通过 proposed policy controls 显式释放。
- event-based pin 缺失时，用 phase TTL 作为兜底，避免 KV 被无限保护。

示例：

```text
SYSTEM_PROMPT:
  phase_ttl_ms = long
  priority = high

USER_QUERY:
  phase_ttl_ms = short
  reuse_scope = SESSION

PARTIAL_PREFILL:
  phase_ttl_ms = very_short
  pin_until_event = tool_returned

RESPONSE:
  admission = skip
```

### 4.5 Semantic Admission

Advanced hints 中的 `semantic_spans` 和 `policy.admission` 用于决定 `skip / report / put`：

```text
skip:
  不上报 UMBP，不保存 KV bytes，只让推理引擎本地按请求生命周期处理

report:
  只上报 key、tier、worker、metadata，用于 match_external_kv() 和 routing

put:
  把 KV bytes 交给 UMBP 托管，用于跨节点、跨 tier 复用
```

推荐默认策略：

- `SYSTEM_PROMPT`：默认 report；高命中或全局共享前缀可 put。
- `USER_QUERY`：session-scoped、短 phase TTL，通常 report，不长期 put。
- `TOOL_OUTPUT`：根据大小、cacheable hint 和命中统计决定；大且低命中时 skip 或 report。
- `PARTIAL_PREFILL`：短 phase TTL，可保护到工具返回。
- `RESPONSE`：默认 skip，不 report、不 put。
- `UNKNOWN`：保守处理，skip 或短 phase TTL report。

### 4.6 Canonical Key 生成与安全复用

Scheduler 不应只根据语义判断复用。Advanced hints 中的 `key_identity` 用于生成 canonical key：

```text
cache_key = hash(
    model_id,
    tokenizer_id,
    kv_layout_version,
    token_chunk_hash,
    session_id,
    reuse_scope,
    phase,
    cache_salt
)
```

关键原则：

- `token_chunk_hash` 证明 token 内容一致。
- `model_id / tokenizer_id / kv_layout_version` 防止不同模型或 KV layout 误复用。
- `session_id / cache_salt / reuse_scope` 控制安全边界。
- `phase` 可以参与 key 或 metadata，但不能单独证明复用正确性。

如果 key_identity 不完整，scheduler 可以允许 `report` 做 routing hint，但不应允许 `put` 或跨 session/global 复用。

---

## 5. Scheduler 调度 UMBP 与推理 Worker

v2 中 scheduler 同时驱动推理 worker 和 UMBP，而不是通过 adapter 间接控制 UMBP：

```text
Scheduler decision
  ├── worker_action:
  │     ├── route request to selected worker
  │     ├── prefill / decode
  │     ├── expose KV layout / ptr for UMBP-owned chunks
  │     ├── report locally-owned KV hashes when using metadata-only mode
  │     └── consume KV fetched from UMBP
  │
  └── umbp_action:
        ├── metadata path:
        │     ├── match  -> UMBP match_external_kv()
        │     ├── report -> UMBP report_external_kv_blocks()
        │     └── revoke -> UMBP revoke_external_kv_blocks()
        │
        └── owned-by-UMBP bytes path:
              ├── put    -> UMBP batch_put_from_ptr()
              ├── get    -> UMBP batch_get_into_ptr()
              ├── policy -> priority / depth / phase TTL / tier hints
              │
              └── proposed policy controls:
                    ├── pin    -> protect key/block until event or phase TTL
                    ├── demote -> move key/block from hot tier to colder tier
                    └── evict  -> release low-value key/block explicitly
```

这里要保持三条边界清晰：

- 推理 worker 负责模型执行、本地 KV cache、KV layout / pointer 暴露，以及消费从 UMBP 取回的 KV。
- UMBP metadata path 只维护 externally-owned KV 的 hash、tier、node 信息，用于 `match_external_kv()` 这类 advisory routing；这些 external KV blocks 不能通过 UMBP 的 owned-key data path 直接读取 bytes。
- UMBP-owned bytes path 使用 `batch_put_from_ptr()` / `batch_get_into_ptr()` 托管和取回 KV bytes；peer 持有真实 slot、page location、tier storage 和 eviction 状态。
- UMBP 不解析 prompt，不理解 `SYSTEM_PROMPT` 或 `TOOL_OUTPUT` 的业务含义。

在这个基础上，`pin`、`demote`、`evict` 可以作为 v2 为 hints 新增的 policy control 接口。它们不是当前源码已经稳定暴露的基础数据面接口，而是把 scheduler 从“只给 priority / phase TTL 建议”推进到“可以显式保护、降级和释放特定 KV”的闭环控制。

这类新增接口建议保持 key/block-level，而不是 semantic-level：

```text
pin(key_or_block, until_event | phase_ttl)
  protect hot or event-critical KV from eviction

demote(key_or_block, target_tier)
  move reusable but cold KV from HBM/DRAM to SSD or remote tier

evict(key_or_block, reason)
  release KV that scheduler has classified as low-value or expired
```

Scheduler 负责把 semantic hints 翻译成这些 key/block-level 控制动作。例如 `PARTIAL_PREFILL` 可以保护到 tool result 返回，`SYSTEM_PROMPT` 可以长期保留在热层，低价值 `RESPONSE` 可以在内存压力下显式 evict。UMBP 仍然只接收 opaque key、block id、tier、phase TTL 和 priority，不需要理解这些 phase 的业务语义。

---

## 6. 端到端例子

假设企业报表 Agent 的一次任务包含：

```text
SYSTEM_PROMPT:  公司统一分析规范，约 3K tokens
USER_QUERY:     “分析华东区 Q2 销售异常”
TOOL_OUTPUT:    SQL 查询结果，约 20K tokens
PARTIAL_PREFILL: 等待 SQL 时已 prefill 的工具无关上下文
RESPONSE:       最终中文报告
```

### 6.1 上层产生并携带 hints

Agent、Harness 或 Agent Orchestrator 可以产生 base hints；Request Frontend 负责把这些 hints 连同请求、tokens 和 KV layout metadata 一起传给 scheduler：

```json
{
  "base": {
    "session_id": "report-task-001",
    "priority": 5,
    "expected_output_tokens": 1024,
    "session_action": "open",
    "session_timeout_ms": 600000
  }
}
```

如果上层有更丰富的语义上下文，也可以提供 advanced hints：

```json
{
  "advanced": {
    "semantic_spans": [
      {
        "token_start": 0,
        "token_end": 3072,
        "phase": "SYSTEM_PROMPT",
        "source": "role:system"
      },
      {
        "token_start": 3072,
        "token_end": 3120,
        "phase": "USER_QUERY",
        "source": "role:user"
      },
      {
        "token_start": 3120,
        "token_end": 23500,
        "phase": "TOOL_OUTPUT",
        "source": "tool_event",
        "cacheable": false
      }
    ],
    "key_identity": {
      "model_id": "llama-...",
      "tokenizer_id": "tok-...",
      "kv_layout_version": "v1",
      "reuse_scope": "SESSION"
    }
  }
}
```

### 6.2 Scheduler 决策

Scheduler 把 base hints 和 semantic spans 转成 admission、routing、tier 和 lifecycle 策略：

```text
SYSTEM_PROMPT:
  - 生成 canonical key
  - admission = report
  - 如果命中率高，升级为 put
  - priority = high
  - phase_ttl_ms = long

USER_QUERY:
  - reuse_scope = SESSION
  - admission = report
  - phase_ttl_ms = short

TOOL_OUTPUT:
  - 如果 cacheable=false 或过大且低命中，admission = skip
  - 如果后续请求或 agent step 会复用，admission = report 或 put

PARTIAL_PREFILL:
  - pin_until_event = tool_returned
  - phase_ttl_ms = very short

RESPONSE:
  - admission = skip
```

### 6.3 UMBP 执行

Scheduler 再把策略落到 UMBP 的 metadata path、bytes path 和 proposed policy controls：

```text
1. match_external_kv(system_prompt_key)
   -> 如果已有 worker 持有系统提示词 KV，优先路由过去。

2. report_external_kv_blocks(system_prompt_key, worker, tier, metadata)
   -> 后续同 session 请求可以通过 UMBP 找到已有 KV。

3. batch_put_from_ptr(system_prompt_key, kv_ptr)
   -> 对高命中系统提示词，交给 UMBP 托管。

4. pin / demote / evict / update phase TTL
   -> 对高价值 KV 做事件级保护，对短期 user query / tool output 到期降级或释放。
```

这个例子的重点是：**scheduler 直接把 hints 转成 UMBP 动作**。UMBP 不需要知道“报表”或“SQL”的业务含义，只执行 key、metadata、tier 和 policy controls。

---

## 7. 落地阶段

### 阶段一：Base Hints + Metadata-only Routing

先支持：

- `session_id`
- `priority`
- `session_timeout_ms`
- `expected_output_tokens`
- `session_action`

Scheduler 做：

- sticky session routing
- UMBP `match_external_kv()`
- UMBP `report_external_kv_blocks()`
- session timeout 到期 revoke metadata
- priority queueing

收益：

- 低风险，不改变 KV bytes 所有权。
- 先获得 Dynamo 风格的 KV locality 和 routing 收益。
- 为后续 policy controls 积累命中率、生命周期和 tier 状态数据。

### 阶段二：Advanced Semantic Admission

加入：

- semantic spans
- phase
- admission `skip / report / put`
- phase-specific TTL / priority

Scheduler 做：

- `SYSTEM_PROMPT` 优先 report/put
- `RESPONSE` 默认 skip
- `TOOL_OUTPUT` 根据 cacheable、大小、命中统计做 admission
- `PARTIAL_PREFILL` 短 phase TTL；event pin 作为 scheduler 侧保护策略

收益：

- 减少 cache 污染。
- 让高价值 KV 更稳定保留。

### 阶段三：UMBP-owned KV 与 Policy Controls

加入：

- 完整 canonical key
- KV layout metadata
- `batch_put_from_ptr()` / `batch_get_into_ptr()`
- tier placement / eviction priority / phase TTL policy hints
- proposed `pin` / `demote` / `evict` policy controls

Scheduler 做：

- 只把高价值 KV 交给 UMBP 托管。
- 根据 priority / phase TTL / phase 影响 HBM/DRAM/SSD 分层和 eviction 顺序。
- 在事件结束或 phase TTL 到期后调用 policy controls，允许 UMBP 降级、回收或撤销 metadata。

收益：

- 支持跨节点、跨 tier KV 复用。
- 减少重复 prefill 和热层 cache 污染。

---

## 8. 设计原则

1. **Scheduler owns policy, UMBP executes.** Scheduler 消费 hints 并做策略决策；UMBP 执行 key、metadata、data movement 和 tier 操作。
2. **Base first, advanced later.** 先落地稳定请求级 hints，再引入 token span 级语义。
3. **Hints are soft.** Scheduler 可以接受、裁剪、忽略或降级 hints，避免上层无限 pin 或污染缓存。
4. **Semantic value is not reuse correctness.** 语义决定缓存价值和生命周期；canonical key 决定能否安全复用。
5. **Extend UMBP at key/block level.** 新增 policy controls 应围绕 opaque key、block id、phase TTL、priority 和 tier，不把 Agent 语义下沉到 UMBP core。

---

## 9. 小结

UMBP Agent Hints v2 的核心变化是：**从 adapter-centric 改为 scheduler-centric**。

Base hints 借鉴 Dynamo，先解决 session affinity、priority queueing、session lifecycle 和 output load estimation。Advanced hints 继承 v1 中的 Sutradhara semantic phase 和 LMCache-style key 管理，让 scheduler 能从 `semantic_spans.phase` 衍生 `phase_ttl_ms`、admission 和 pin 等缓存策略。

最终形态是：

```text
Agent / Harness / Agent Orchestrator 产生 hints
Request Frontend 携带 hints 进入 serving 层
Scheduler 消费 hints 并决策缓存策略
UMBP 执行 key-based routing、storage、tiering 和 policy controls
```

这样既贴合现有 scheduler 架构，也为 UMBP 补齐 agent-hint-driven cache optimization 所需的控制面，同时避免让 UMBP core 承担 Agent 语义理解。
