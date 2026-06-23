# UMBP Agent Hints 方案 v2：Scheduler 驱动的语义缓存策略

## 1. 背景与目标

v1 方案把核心组件设计成 **Semantic KV Adapter**：adapter 位于推理框架和 UMBP 之间，负责把 Agent 语义翻译成 `semantic spans`、`canonical key`、`priority / TTL` 和 `skip / report / put` 决策，再通过 UMBP API 与 UMBP 交互。

v2 方案调整架构边界：**不再把 adapter 作为 UMBP 的主要交互层，而是让 scheduler 成为 agent hints 的消费方和策略决策方**。这更接近 NVIDIA Dynamo 的思路：上层通过请求级 hints 暴露 session、priority、输出长度、生命周期等信息，serving 层 scheduler/router 根据这些 hints 做路由、排队和 KV cache 策略优化。

v2 的目标是：

1. **贴合现有软件架构**：系统已有 scheduler，缓存策略应由 scheduler 统一决策，而不是新增 adapter 分裂控制面。
2. **分层接收 hints**：先支持稳定、容易生产的 base hints，再扩展到包含语义和 key 管理的 advanced hints。
3. **让 UMBP 保持执行层定位**：UMBP 不理解 Agent 语义，只接收 scheduler 下发的 key、metadata 和策略动作。
4. **围绕 semantic-driven cache policy optimization**：语义用于决定缓存价值、生命周期、准入和 eviction/pin 策略；复用安全仍由 token chunk、model、tokenizer、KV layout 和隔离字段保证。

---

## 2. 总体架构

v2 的核心链路是：

```text
+------------------------------------------------------------+
| Agent / Workflow Orchestrator / Harness                    |
|                                                            |
| base hints: session_id / workflow_id / ttl / priority      |
| advanced:   semantic spans / reuse_scope / key fields      |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
| LLM Provider / Engine Connector                            |
|                                                            |
| attach hints with request, tokens, and KV layout metadata   |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
| Scheduler                                                  |
|                                                            |
| decide route / prefetch / report / put / revoke / tiering  |
| decide TTL / priority / pin / eviction                     |
+-----------------------------+------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+----------------------------+   +----------------------------+
| Inference Workers          |   | UMBP                       |
| SGLang / vLLM / others     |   |                            |
|                            |   |                            |
| - prefill / decode         |   | - match_external_kv()      |
| - local KV cache           |   | - report / revoke metadata |
| - expose KV layout / ptr   |   | - RoutePut / RouteGet      |
| - consume fetched KV       |   | - batch_put/get ptr APIs   |
|                            |   | - HBM / DRAM / SSD tier    |
+----------------------------+   +----------------------------+
              ^                               ^
              |                               |
              +------ KV metadata / ptr / transfer -----------+
```

这个架构中，scheduler 是策略中枢：

- 它像 Dynamo Router 一样消费请求级 hints，做 routing、queueing、KV locality 和 session affinity。
- 它也消费 advanced semantic hints，做更细粒度的 KV admission、TTL、priority、pin 和 eviction/tier 策略。
- 推理 worker 和 UMBP 是 scheduler 下的两个同级执行子系统：worker 负责 prefill/decode 和本地 KV，UMBP 负责跨节点、跨 tier 的 KV 查询、搬运和托管。
- UMBP 不直接解析 Agent 语义，只执行 scheduler 下发的 key、metadata 和缓存策略动作。

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
  request_id              # 单次请求标识，用于 tracing，不作为 KV identity
  session_id              # 会话标识，用于 sticky routing、session-scoped reuse、生命周期管理
  workflow_id             # workflow / agent run 标识，用于关联同一任务轨迹
  tenant_id               # 租户隔离
  priority                # 请求或缓存保留优先级
  strict_priority         # 可选，强优先级层级
  ttl_ms                  # 请求相关 KV 的默认生命周期
  expected_output_tokens  # 类似 Dynamo osl，用于输出 KV block 负载估计
  expected_next_call_ms   # 类似 iat，用于估计下一轮请求间隔
  total_expected_calls    # 类似 total_requests，用于估计 workflow 剩余生命周期
  session_action          # open / bind / close
  session_timeout_ms      # session inactivity timeout
```

这些字段不要求上层提供完整语义，只要求提供 serving 层容易理解的信息：

- `session_id` 让 scheduler 做 sticky session routing，减少同一会话 KV 在 worker 间漂移。
- `priority` 影响 scheduler queue ordering、UMBP eviction/tier 策略。
- `ttl_ms` 告诉 scheduler 某类 KV 多久后可以降级、撤销 metadata 或回收。
- `expected_output_tokens` 帮助 scheduler 估算 decode 阶段的 KV 增长和 worker 未来负载。
- `session_action` / `session_timeout_ms` 支持 session open/close、subagent 生命周期和自动清理。

Base hints 的特点是：**即使没有 token span 级语义，也能带来 routing locality、session isolation、priority queueing 和生命周期控制收益**。

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
    pin_until_event      # tool_returned / workflow_step_done / session_close
    admission            # skip / report / put，可由上层建议，也可由 scheduler 决定
```

Advanced hints 的核心作用：

- **语义价值识别**：识别 `SYSTEM_PROMPT`、`TOOL_OUTPUT`、`RESPONSE` 等不同 KV 的复用价值。
- **语义生命周期策略**：把 phase 转成 TTL、priority、pin。
- **语义安全复用**：把 phase、reuse scope、tenant/session 与 LMCache-style token chunk key 结合，避免“语义相似”导致错复用。
- **语义准入控制**：决定 chunk 是 `skip`、`report` 还是 `put`。

Advanced hints 不是一开始必须全部具备。没有显式 semantic spans 时，scheduler 可以根据 message role、tool event 和默认规则生成保守 phase；没有完整 key_identity 时，只允许 metadata-only routing，不进入 UMBP-owned KV bytes 托管。

---

## 4. Scheduler 如何消费 Hints

Scheduler 是 v2 的核心策略层。它不只是排队组件，而是 agent hints 到 UMBP 缓存动作的决策中心。

### 4.1 Routing 与 Session Affinity

Base hints 中的 `session_id`、`workflow_id` 和 `tenant_id` 用于 routing：

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

Base hints 中的 `priority` / `strict_priority` 用于 scheduler 排队：

```text
queue_order = (strict_priority, policy_score(priority, arrival_time, token_cost))
```

可能策略：

- 高优先级请求先调度。
- 长 prefill、低优先级请求可延后或降级。
- 对 latency-sensitive 请求提高 routing 和 cache retrieval 优先级。

`priority` 也可以传给 UMBP 作为 eviction/tier policy hint：

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

### 4.4 TTL 与 Lifecycle

Base `ttl_ms` 和 advanced `phase_ttl_ms` 都是生命周期 hint。

Scheduler 可以做：

- metadata-only 模式下，TTL 到期后 revoke external KV metadata。
- UMBP-owned 模式下，TTL 到期后降低 priority、迁移到冷 tier 或释放。
- event-based pin 缺失时，用 TTL 作为兜底，避免无限 pin。

示例：

```text
SYSTEM_PROMPT:
  ttl_ms = long
  priority = high

USER_QUERY:
  ttl_ms = short
  reuse_scope = SESSION

PARTIAL_PREFILL:
  ttl_ms = very_short
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
- `USER_QUERY`：session-scoped、短 TTL，通常 report，不长期 put。
- `TOOL_OUTPUT`：根据大小、cacheable hint 和命中统计决定；大且低命中时 skip 或 report。
- `PARTIAL_PREFILL`：短 TTL，可 pin 到工具返回。
- `RESPONSE`：默认 skip-save，不 report、不 put。
- `UNKNOWN`：保守处理，skip 或短 TTL report。

### 4.6 Canonical Key 生成与安全复用

Scheduler 不应只根据语义判断复用。Advanced hints 中的 `key_identity` 用于生成 canonical key：

```text
cache_key = hash(
    model_id,
    tokenizer_id,
    kv_layout_version,
    token_chunk_hash,
    tenant_id,
    session_id or workflow_id,
    reuse_scope,
    phase,
    cache_salt
)
```

关键原则：

- `token_chunk_hash` 证明 token 内容一致。
- `model_id / tokenizer_id / kv_layout_version` 防止不同模型或 KV layout 误复用。
- `tenant_id / cache_salt / reuse_scope` 控制安全边界。
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
  │     ├── expose KV layout / ptr for reusable chunks
  │     └── consume KV fetched from UMBP
  │
  └── umbp_action:
        ├── match       -> UMBP match_external_kv()
        ├── report      -> UMBP report_external_kv_blocks()
        ├── revoke      -> UMBP revoke external metadata
        ├── put         -> UMBP batch_put_from_ptr()
        ├── get         -> UMBP batch_get_into_ptr()
        ├── pin         -> UMBP mark protected / high priority until event or TTL
        ├── demote      -> UMBP move HBM/DRAM -> SSD or cold tier
        └── evict       -> UMBP release low-value KV
```

两边的职责边界保持清晰：

- 推理 worker 负责模型执行、本地 KV cache、KV layout / pointer 暴露，以及消费从 UMBP 取回的 KV。
- UMBP 根据 opaque key 做查找、搬运、托管、分层和回收。
- UMBP 不解析 prompt，不理解 `SYSTEM_PROMPT` 或 `TOOL_OUTPUT` 的业务含义。

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

### 6.1 上层产生 hints

Harness / Orchestrator / LLM Provider 可以产生 base hints：

```json
{
  "base": {
    "session_id": "report-task-001",
    "workflow_id": "sales-report-agent",
    "tenant_id": "tenant-a",
    "priority": 5,
    "ttl_ms": 300000,
    "expected_output_tokens": 1024,
    "session_action": "open",
    "session_timeout_ms": 600000
  }
}
```

如果上层有更丰富语义，也可以提供 advanced hints：

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

Scheduler 根据 hints 做策略：

```text
SYSTEM_PROMPT:
  - 生成 canonical key
  - admission = report
  - 如果命中率高，升级为 put
  - priority = high
  - ttl = long

USER_QUERY:
  - reuse_scope = SESSION
  - admission = report
  - ttl = short

TOOL_OUTPUT:
  - 如果 cacheable=false 或过大且低命中，admission = skip
  - 如果后续 workflow 会复用，admission = report 或 put

PARTIAL_PREFILL:
  - pin_until_event = tool_returned
  - ttl = very short

RESPONSE:
  - admission = skip
```

### 6.3 UMBP 执行

Scheduler 调用 UMBP：

```text
1. match_external_kv(system_prompt_key)
   -> 如果已有 worker 持有系统提示词 KV，优先路由过去。

2. report_external_kv_blocks(system_prompt_key, worker, tier, metadata)
   -> 后续同 workflow 请求可以通过 UMBP 找到已有 KV。

3. batch_put_from_ptr(system_prompt_key, kv_ptr)
   -> 对高命中系统提示词，交给 UMBP 托管。

4. revoke 或 TTL 到期降级
   -> 对短期 user query / tool output，避免长期污染热层。
```

这个例子的重点是：**scheduler 直接把 hints 转成 UMBP 动作**。UMBP 不需要知道“报表”或“SQL”的业务含义，只执行 key、metadata 和 policy。

---

## 7. 与 v1 的差异

| 维度 | v1：Semantic KV Adapter | v2：Scheduler-driven Agent Hints |
|---|---|---|
| 策略中枢 | Adapter | Scheduler |
| 与 UMBP 交互 | Adapter 调 UMBP API | Scheduler 直接调 UMBP API |
| 架构风格 | 更像 LMCache connector / translation layer | 更像 Dynamo router/scheduler consuming hints |
| Hint 分层 | 主要围绕 semantic spans、key、admission | base + advanced 两层 |
| Base 能力 | 没有单独突出 | session、TTL、priority、OSL、workflow lifecycle |
| Advanced 能力 | Sutradhara phase + LMCache key | 继承 v1 语义和 key 管理，但由 scheduler 消费 |
| UMBP 定位 | Adapter 背后的 KV 执行层 | 与推理 worker 平级，都是 scheduler 调度的执行子系统 |

v2 不否定 v1 的语义和 key 管理内容，而是改变控制面位置：**语义和 key 管理仍然需要，但不再由单独 adapter 主导，而是并入 scheduler 的缓存策略决策。**

---

## 8. 落地阶段

### 阶段一：Base Hints + Metadata-only Routing

先支持：

- `session_id`
- `tenant_id`
- `priority`
- `ttl_ms`
- `expected_output_tokens`
- `session_action`

Scheduler 做：

- sticky session routing
- UMBP `match_external_kv()`
- UMBP `report_external_kv_blocks()`
- TTL 到期 revoke metadata
- priority queueing

收益：

- 低风险，不改变 KV bytes 所有权。
- 先获得 Dynamo 风格的 KV locality 和 routing 收益。

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
- `PARTIAL_PREFILL` 短 TTL / pin

收益：

- 减少 cache 污染。
- 让高价值 KV 更稳定保留。

### 阶段三：UMBP-owned KV 与 Tier Policy

加入：

- 完整 canonical key
- KV layout metadata
- `batch_put_from_ptr()` / `batch_get_into_ptr()`
- tier placement / demote / pin

Scheduler 做：

- 只把高价值 KV 交给 UMBP 托管。
- 根据 priority / TTL / phase 做 HBM/DRAM/SSD 分层。
- 在事件结束或 TTL 到期后解除 pin、降级或释放。

收益：

- 支持跨节点、跨 tier KV 复用。
- 减少重复 prefill 和热层 cache 污染。

---

## 9. 设计原则

1. **Scheduler owns policy, UMBP executes.** Scheduler 消费 hints 并做策略决策；UMBP 执行 key、metadata 和 tier 操作。
2. **Base first, advanced later.** 先落地稳定请求级 hints，再引入 token span 级语义。
3. **Hints are soft.** Scheduler 可以接受、裁剪、忽略或降级 hints，避免上层无限 pin 或污染缓存。
4. **Semantic value is not reuse correctness.** 语义决定缓存价值和生命周期；canonical key 决定能否安全复用。
5. **No semantic logic in UMBP core.** UMBP core 不解析 prompt，不理解业务语义，只消费 scheduler 下发的 opaque key 和轻量 metadata。

---

## 10. 小结

UMBP Agent Hints v2 的核心变化是：**从 adapter-centric 改为 scheduler-centric**。

Base hints 借鉴 Dynamo，先解决 session affinity、priority queueing、TTL 生命周期和 output load estimation。Advanced hints 继承 v1 中的 Sutradhara semantic phase 和 LMCache-style key 管理，让 scheduler 能做 semantic-driven cache policy optimization。

最终形态是：

```text
Agent / Orchestrator 产生 hints
Scheduler 消费 hints 并决策缓存策略
UMBP 执行 key-based routing、storage、tiering、eviction
```

这样既贴合现有 scheduler 架构，也避免让 UMBP core 承担 Agent 语义理解。
