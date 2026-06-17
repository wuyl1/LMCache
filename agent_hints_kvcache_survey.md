# Agent Hints 与 KV Cache 系统调研及设计：Sutradhara、LMCache、Dynamo 对比与 UMBP 方案

## 0. 阅读范围与判断口径

本文使用两类资料来避免把概念推断和代码事实混在一起：

- **外部系统资料**：Sutradhara、NVIDIA Dynamo 的机制参考用户提供的调研报告 `report.md`，并补充核对了公开资料：Sutradhara arXiv HTML、Microsoft Research 发布页、NVIDIA Dynamo 官方 Agents 文档、NeMo Agent Toolkit 的 Dynamo LLM provider 源码/文档、Dynamo Router README，以及 LMCache arXiv HTML。[R1][R2][R3][R4][R5][R6][R7][R8][R9]
- **LMCache 当前实现**：LMCache 相关判断来自当前仓库代码，包括 vLLM adapter、cache key、token database、cache engine、multiprocess connector、storage backend 和配置文件。[C1][C2][C3][C4][C5][C6][C7][C8]

---

## 1. 核心结论

Sutradhara、LMCache、NVIDIA Dynamo 都在解决 Agent 工作流里的 KV Cache 复用问题，但切入层次不同。

- **Sutradhara** 把语义标记送进推理引擎内部，让引擎知道哪些 KV 来自系统提示词、工具输出、最终响应或 partial prefill。[R1]
- **LMCache** 当前没有原生 `agent_hints`，它更像外置 KV Cache 管理层，通过 `lookup`、`pin`、`store/retrieve`、`move`、`clear`、`compress` 等机制管理 KV 生命周期。[C3][C4]
- **Dynamo** 通过请求级 `nvext.agent_hints` 做 KV 感知路由，让相关请求更容易被路由到已有 KV 前缀的 Worker。[R2]

一句话概括：

> Sutradhara 做语义感知缓存管理，LMCache 做外置 KV 管理，Dynamo 做分布式 KV 感知路由。

---

## 2. Sutradhara：语义标记驱动的 Agent Hints

### 2.1 实现方式

Sutradhara 的核心做法，是让 Agent 编排器把 prompt 的内部结构告诉推理引擎。根据公开论文和 Microsoft Research 的介绍，它提供了一组 thin API，用来把编排器侧的信息传给引擎；这些 API 支持工具感知的 prompt splitting、流式工具调度，以及由语义提示驱动的缓存管理。[R1][R4][R5]

论文列出的核心 API 包括：

- `submit_partial_prefill()`：提交工具无关的 prompt slice，使引擎可在工具执行期间提前 prefill。
- `extend_prefill()`：工具输出返回后，把工具相关后缀拼接到已固定的 partial prefill context。
- `register_streaming_callback()`：逐 token 接收 decode 输出，便于工具调用 JSON 一完成就被派发。
- `tag_kv_blocks()`：给 KV block 附加语义 hint，例如 `system_prompt`、`response`。
- `set_reuse_priority()`：给 KV block 设置复用/固定优先级。[R4]

在语义缓存管理上，Sutradhara 使用几类标签描述 KV block 的来源：

- `SYSTEM_PROMPT`
- `USER_QUERY`
- `TOOL_OUTPUT`
- `RESPONSE`
- `PARTIAL_PREFILL`

这些标签不是只用于记录日志的 metadata，而是会进入引擎的缓存管理决策，例如驱逐优先级、复用优先级和 partial prefill 生命周期。[R1][R4]

### 2.2 示例

例如，一个 Agent prompt 可以拆成几段：

```text
0-800     system prompt + tool schema
800-950   user query
950-1300  tool output
1300-1400 assistant response
```

Sutradhara 会把这些 token span 标成对应语义类型：

```text
0-800     SYSTEM_PROMPT
800-950   USER_QUERY
950-1300  TOOL_OUTPUT
1300-1400 RESPONSE
```

引擎拿到这些标签后，就能区分不同 KV block 的复用价值：

- `SYSTEM_PROMPT`：复用价值高，最后驱逐。
- `TOOL_OUTPUT`：可能被后续轮次复用，中等优先级。
- `RESPONSE`：通常一次性内容，优先驱逐。
- `PARTIAL_PREFILL`：工具执行期间临时强保护。

这里的优先级关系来自报告和 Sutradhara 论文中的描述：容量不足时，引擎先驱逐低优先级 block；同一优先级层内再用 LRU 作为 tie-breaker。论文给出的驱逐顺序是 `RESPONSE -> TOOL_OUTPUT -> USER_QUERY -> SYSTEM_PROMPT -> PARTIAL_PREFILL`，越靠后越晚驱逐。[R1][R4]

### 2.3 优点

- 能区分不同 prompt 片段的复用价值。
- 能做语义感知驱逐。
- 能支持 partial prefill 与工具执行重叠。
- 对多轮工具调用 Agent 很有针对性。
- Microsoft Research 发布页还总结了其动机：工具调用占 FTR 延迟的显著比例、跨迭代存在上下文复用但 KV 命中率下降、顺序编排浪费请求内并行机会；Sutradhara 通过 co-design 缓解这些问题。[R5]

### 2.4 缺点

- 需要推理引擎暴露内部接口。
- 与 scheduler、KV block manager、eviction policy 耦合较强。
- 工程改造成本高。
- 跨引擎兼容性较弱。

---

## 3. LMCache：外置 KV Cache 管理层

### 3.1 当前是否有 Agent Hints

当前 LMCache 没有原生 `agent_hints` 机制。代码里可以看到 `request_configs`、`cache_salt`、`lmcache.tag.*`、`layout_hints` 等机制，但没有统一的 `agent_hints` 字段，也没有 Sutradhara 风格的语义标签 API。[C1][C2][C5]

它没有：

- Dynamo 风格的 `nvext.agent_hints`
- Sutradhara 风格的 `SYSTEM_PROMPT / TOOL_OUTPUT / RESPONSE`
- 基于语义标签的驱逐优先级
- token span 级语义 metadata

LMCache 当前提供的是更通用的 KV Cache 管理能力：

- `lookup`
- `store / retrieve`
- `pin / unpin`
- `move`
- `clear`
- `compress / decompress`
- `cache_salt`
- `lmcache.tag.*`
- `lmcache.skip_save`
- 分层存储

这些能力分布在不同模块中：`lookup`、`move`、`compress`、`decompress`、`clear` 位于 `LMCacheEngine`；`pin/unpin` 由 storage backend 和 lookup pin 路径实现；`cache_salt` 在 MP key 中作为缓存身份的一部分；`lmcache.tag.*` 在 `CacheEngineKey` 中进入 `tags`。[C2][C3][C4][C6]

### 3.2 LMCache 如何复用 Agent 前缀

LMCache 的复用条件可以简化理解为：

```text
token 前缀完全一致 + chunk 边界对齐
```

它不理解某段 token 是 `SYSTEM_PROMPT`。但 system prompt 通常稳定地位于 prompt 前部，因此很容易自然形成精确前缀命中。LMCache 的 token 处理路径会按 chunk 生成 prefix hash，再构造 `CacheEngineKey`；`lookup` 返回的是连续前缀命中的 token 数。[C2][C3]

### 3.3 示例：长系统提示词复用

假设 LMCache 的 chunk size 是 `256 tokens`。

第一次 Agent 请求：

```text
0-799    system prompt + tool schema
800-999  user query
```

第一次没有缓存，vLLM 正常 prefill。prefill 完成后，LMCache connector 在 chunk 边界保存 KV：

```text
chunk 0: token 0-255
chunk 1: token 256-511
chunk 2: token 512-767
```

第二次 Agent 请求：

```text
0-799     system prompt + tool schema     完全相同
800-999   user query                      完全相同
1000-1199 tool output                     新增内容
```

LMCache lookup 发现：

```text
token 0-255   命中
token 256-511 命中
token 512-767 命中
token 768-... 未命中
```

于是 LMCache 返回：前 `768 tokens` 可以从外部 KV cache 加载。

后续流程：

1. 命中的 KV chunk 被 `pin`，避免 retrieve 前被淘汰。
2. vLLM 为这些 token 分配 paged KV block。
3. LMCache worker 将 KV retrieve 回 vLLM 的 KV block。
4. `768` token 之后的新内容由 vLLM 正常 prefill。
5. 请求完成后释放 pin，并可能保存新计算出的 chunk。

这个例子的关键点是：LMCache 复用的是“精确 token 前缀”，不是因为它知道这段内容是 `SYSTEM_PROMPT`。

代码依据：`ChunkedTokenDatabase` 按 `chunk_size` 切分 token 并生成 prefix hash；`LMCacheEngine.lookup()` 对这些 key 做连续前缀命中检查；vLLM MP connector 根据 LMCache 命中 token 数和 vLLM 已计算 token 数决定需要 retrieve 的范围。[C2][C3][C5]

### 3.4 `lmcache.tag.*`

`lmcache.tag.*` 是 LMCache 中最接近 hint-like metadata 的机制，但它的作用边界要说清楚。

在 vLLM v1 adapter 中，`kv_transfer_params` 中所有 `lmcache.` 前缀字段会被提取为 `request_configs`；随后 `CacheEngineKey` 只把 `lmcache.tag.*` 前缀字段抽取到 `tags`，并把 `tags` 纳入 key 的 hash/equality。[C1][C2]

例如请求携带：

```json
{
  "lmcache.tag.agent": "planner",
  "lmcache.tag.phase": "system_prompt",
  "lmcache.tag.tenant": "alice"
}
```

这些 tag 会进入 cache key，成为缓存身份的一部分。

例如同一段 token：

```text
You are an AI coding assistant.
Available tools: ...
```

请求 A：

```text
lmcache.tag.phase = system_prompt
```

请求 B：

```text
lmcache.tag.phase = tool_output
```

即使 token 完全一样，cache key 也会不同：

```text
hash(tokens) + phase=system_prompt
hash(tokens) + phase=tool_output
```

结果是：

- A 保存的 KV 只有同样带 `phase=system_prompt` 的请求能命中。
- B 保存的 KV 只有同样带 `phase=tool_output` 的请求能命中。
- A 和 B 不会互相复用。

但 `lmcache.tag.phase=system_prompt` 不会让 LMCache 自动提高该缓存的保留优先级。它控制的是“能不能共享”，不是“谁更值得保留”。

这是因为当前 `CacheEngineKey` 只把 tag 放入缓存身份；本仓库中未见 storage manager 或 eviction policy 对 `lmcache.tag.phase` 做语义优先级解释。[C2][C4]

### 3.5 `cache_salt`

`cache_salt` 用于缓存身份隔离。

例如：

```text
tenant A: cache_salt = tenant-a
tenant B: cache_salt = tenant-b
```

即使两个租户 prompt 完全相同，LMCache 也会把它们当作不同缓存空间。

适用场景：

- 多租户隔离
- 按租户统计缓存使用
- 按租户做 quota
- 按租户做 isolated eviction

代码依据：MP 路径的 `IPCCacheServerKey` 注释明确说明 `request_id` 不属于缓存身份，而 `cache_salt` 属于缓存身份；分布式 L2 adapter 和 isolated LRU 也按 `cache_salt` 统计使用量或执行隔离驱逐。[C6][C7]

### 3.6 `lmcache.skip_save`

如果某个 Agent 请求是一次性请求，不希望污染缓存，可以传入：

```text
lmcache.skip_save = True
```

LMCache adapter 会读取这个字段，并在保存路径中跳过该请求。

它可以粗粒度表达“低复用价值内容不缓存”，但不是 token span 级语义标签。

代码依据：vLLM v1 adapter 在生成请求 metadata 时读取 `tracker.request_configs` 中的 `lmcache.skip_save`，并把它并入 `skip_save` 判断。[C1]

### 3.7 优点

- 与推理引擎松耦合。
- 支持不同存储层和 backend。
- 适合生产环境做外部 KV 管理。
- 支持精确 lookup、pin、move、clear、compress。
- 可作为 Dynamo、llm-d、AIBrix 等系统的 KV 基础设施。

### 3.8 缺点

- 当前没有原生 Agent Hints。
- 不理解 `SYSTEM_PROMPT / RESPONSE / TOOL_OUTPUT` 等语义类型。
- 不能基于语义优先级驱逐。
- 复用主要依赖 token 前缀精确匹配。
- `lmcache.tag.*` 只影响 cache key，不影响 eviction priority。

---

## 4. NVIDIA Dynamo：请求级 Agent Hints 与 KV 感知路由

### 4.1 实现方式

Dynamo 的重点不是 token span 级语义标记，而是请求级路由提示。NVIDIA Dynamo 官方 Agents 文档说明，agent-facing metadata 位于 OpenAI-compatible request body 的 `nvext` 下，包括 `agent_context` 和 `agent_hints`；Dynamo 使用这些元数据做 telemetry、routing hints 和 backend-specific cache behavior。[R6]

这些信息通过请求体扩展字段传递：

```text
nvext.agent_hints
```

典型字段包括：

- `prefix_id`
- `total_requests`
- `osl`
- `iat`
- `latency_sensitivity`
- `priority`

这些字段来自用户提供报告、NVIDIA Dynamo Agents 文档，以及 NeMo Agent Toolkit 的 Dynamo LLM provider/GitHub 示例。[R2][R6][R7][R8]

### 4.2 字段含义

- `prefix_id`：同一 workflow 或共享前缀标识，用于缓存亲和路由。
- `total_requests`：预计剩余 LLM 调用次数。
- `osl`：预测输出长度。
- `iat`：预测请求到达间隔。
- `latency_sensitivity`：延迟敏感度。
- `priority`：调度优先级。Dynamo 官方 Agents 文档说明 `priority` 可用于 engine queue ordering 和 KV cache eviction；NeMo Agent Toolkit 中通常按 `max_sensitivity - latency_sensitivity` 计算，数值越低优先级越高。[R6][R7]

需要注意的是，Dynamo 官方 Agents 文档还列出一些更偏 serving/runtime 的字段，例如 `speculative_prefill`，并把 `program_id`、`context_type` 标为 planned。这说明 Dynamo 的 Agent Hints 正在向更丰富的 agentic serving metadata 扩展；但 `context_type` 这类语义类型在文档中仍是 planned，并不是本文讨论的 Sutradhara 式、已落地的 token span 级语义标签。[R6]

### 4.3 示例

一个 Agent workflow 预计会连续调用 LLM 6 次：

```json
{
  "nvext": {
    "agent_hints": {
      "prefix_id": "agent-run-123",
      "total_requests": 6,
      "osl": 128,
      "iat": 250,
      "latency_sensitivity": 5.0,
      "priority": 1
    }
  }
}
```

第一次请求被路由到 Worker A，Worker A 计算并缓存了该 workflow 的长系统提示词 KV。

第二次请求带相同 `prefix_id`。Dynamo 的 KV Router 发现 Worker A 已有相关前缀 KV，于是倾向继续路由到 Worker A，而不是按轮询发给 Worker B。

结果是：

- 提升 KV cache 局部性。
- 减少冷 miss。
- 降低 TTFT。
- 在多 Worker 场景中提升吞吐。

该路由逻辑来自报告对 Dynamo KV Router、`prefix_id`、请求优先级和缓存亲和路由的总结，也可以从 Dynamo Router README 得到印证：KV Router 会评估 worker 的 prefill/decode 成本，并利用 KV cache overlap 减少重复计算；`--router-queue-threshold` 等配置还会启用基于 `nvext.agent_hints.priority` 的优先级调度。[R2][R3][R8]

### 4.4 优点

- 适合多 Worker、多节点部署。
- 请求级 hints 易于通过 API 传递。
- 能提升 KV 局部性。
- 可结合负载、优先级和缓存命中做路由决策。
- 对生产级分布式推理服务友好。

### 4.5 缺点

- 主要是请求级 hints，不是 token span 级 hints。
- 不直接解决 `SYSTEM_PROMPT` 与 `RESPONSE` 的 block 级驱逐优先级。
- 依赖路由器和集群调度基础设施。
- hint 准确性依赖上层预测。

---

## 5. 三者对比

把三者放在一起看，可以看到它们并不是互相替代的关系，而是分别站在不同层次上优化 KV Cache：

| 维度 | Sutradhara | LMCache | Dynamo |
|---|---|---|---|
| 架构类型 | Orchestrator-Engine 协同设计 | 外置 KV Cache 管理层 | 分布式推理服务与 KV 感知路由 |
| Hint 形式 | token span 级语义标记 | 无原生 agent_hints；支持 cache metadata | 请求级 `nvext.agent_hints` |
| 典型字段 | `SYSTEM_PROMPT`、`TOOL_OUTPUT`、`RESPONSE`、`PARTIAL_PREFILL` | `cache_salt`、`lmcache.tag.*`、`lmcache.skip_save`、`pin` | `prefix_id`、`total_requests`、`osl`、`iat`、`latency_sensitivity`、`priority` |
| 消费位置 | 引擎 scheduler、KV manager、eviction policy | LMCache key、storage manager、lookup/store/retrieve、backend | Router、KV Router、调度层 |
| 是否理解语义 | 是 | 否，当前只把 tag 当 key metadata | 部分，请求级语义 |
| 是否影响 eviction priority | 是 | 当前不原生支持语义优先级 | 间接影响生命周期和路由 |
| 是否适合多 Worker 路由 | 不是重点 | 可提供 lookup 信息给外部路由器 | 是核心能力 |
| 工程耦合 | 高 | 低 | 中 |
| 主要优势 | 语义表达强，优化潜力大 | 兼容性强，KV 管理能力完整 | 分布式路由效果强 |
| 主要局限 | 改造引擎成本高 | 不原生理解 Agent 语义 | 不做细粒度语义 KV 管理 |

### 5.1 性能收益图引用

> 注意：以下图片直接引用论文或官方公开资料中的原图。不同系统的硬件、模型、负载、baseline 和指标定义并不完全一致，因此这些图适合说明机制带来的收益方向和量级，不适合作为严格横向 benchmark。

#### 5.1.1 Sutradhara：论文 Figure 8 / 10 / 11

Sutradhara 论文 Figure 8 给出 serving capacity curves：横轴是 p50/p90 FTR 与 E2E 延迟，纵轴是 ingest load；曲线越靠左上角越好。论文结论是：在相同 p50 FTR 延迟下，Sutradhara 最多可承载 77% 更高负载；固定负载下，p50 FTR 最多降低 15%，p90 FTR 最多降低 11%，p90 E2E 最多降低 9%。[R4][R5]

![Sutradhara Figure 8: serving capacity curves](https://arxiv.org/html/2601.12967v3/x8.png)

论文 Figure 10 进一步拆解 FTR 延迟来源，把代表性请求的 FTR 分为 critical path tool time、prefill time 和 decode time。它说明 Sutradhara 的收益不是单纯来自“缓存命中更多”，还来自流式工具调度降低 critical path tool time，以及 partial prefill / 更高命中率降低 prefill time。[R4]

![Sutradhara Figure 10: FTR latency breakdown](https://arxiv.org/html/2601.12967v3/x10.png)

论文 Figure 11 展示 inter-request 与 intra-request cache hit rate。论文文本说明，全局 cache hit rate 从 21.8% 提升到 44.6%；原因是系统提示词等共享前缀更不容易被驱逐，同时 partial prefill 中包含的前序工具结果也更容易被后续 iteration 复用。[R4]

![Sutradhara Figure 11: cache hit rate analysis](https://arxiv.org/html/2601.12967v3/x11.png)

#### 5.1.2 Dynamo KV Router：官方资料边界

报告中整理的 Dynamo KV Router 结果显示，在 Mooncake Tool Agent Traces 类场景下，KV-aware routing 相比轮询路由显著降低 TTFT 和端到端延迟。[R2] 由于当前可核验的 Dynamo Router README 主要提供机制和配置说明，而不是可直接嵌入的论文性能图，本文不为 Dynamo 额外绘制图表。

说明：

- `TTFT 平均值约 20.4x`、`TTFT P99 约 15.9x`、`E2E 平均值约 4.3x`、`E2E P99 约 3.8x` 来自用户提供报告中对 Dynamo KV Router benchmark 的整理。[R2]
- Dynamo Router README 说明 KV Router 会利用 KV cache overlap 和 worker 的 prefill/decode cost 做路由，减少重复 prefill；这解释了为什么长系统提示词、Agent 多轮前缀复用场景下收益明显。[R8]
- Dynamo 的 `nvext.agent_hints` 和 `nvext.cache_control` 可进一步传递请求优先级、输出长度估计、prefix/session 相关信息，辅助路由和缓存生命周期控制。[R6][R7]
- 没有加入额外图表，是为了避免把报告数值重新画成本文自制图。

#### 5.1.3 LMCache：论文 Figure 8

LMCache 论文 Figure 8 对比了 basic vLLM、basic vLLM CPU offloading、两个商业方案和 LMCache。论文图注说明，LMCache 有 1.9-8.1x 更小 TTFT，并支持 2.3-14x 更高 inference throughput；论文摘要和评估部分还总结其在多类设置下可达到最高 15x 吞吐量提升、至少 2x 延迟降低。[R9]

![LMCache Figure 8: TTFT and throughput comparison](https://arxiv.org/html/2510.09665/x8.png)

说明：

- 这里引用的是 LMCache 论文原图，不是本文在本地仓库重新复现实验的结果。[R9]
- 结合当前代码看，LMCache 的性能收益主要来自外置 KV 管理：`lookup` 发现前缀命中，`pin` 防止 retrieve 前淘汰，`retrieve` 把 KV 加载回引擎，`move/clear/compress` 支持缓存迁移、清理和存储优化。[C3][C4][C5]
- 这些收益不依赖 LMCache 原生理解 `SYSTEM_PROMPT`。LMCache 当前复用的是精确 token 前缀；系统提示词能受益，是因为它通常稳定地位于 prompt 前缀。[C2][C3]

#### 5.1.4 三类机制的收益来源对比

| 系统 | 主要性能收益来源 | 适合场景 | 注意事项 |
|---|---|---|---|
| Sutradhara | 语义标记驱动的优先级驱逐；partial prefill 与工具执行重叠 | 多轮工具调用、prompt 结构明确的 Agent | 需要引擎和编排器协同改造 |
| Dynamo | KV-aware routing；`prefix_id` / `priority` / `osl` 等请求级 hints | 多 Worker、多节点、分布式推理服务 | 主要优化路由层，不是 token span 语义标签 |
| LMCache | 外置 KV lookup/retrieve/store；pin；分层存储与迁移压缩 | 需要跨请求、跨层、跨 backend 管理 KV 的生产系统 | 当前不原生理解 `SYSTEM_PROMPT` / `RESPONSE` 语义 |

---

## 6. LMCache 能否结合 Sutradhara 的优点

可以，但这不是当前已有功能，需要在 LMCache 现有外置 KV 管理能力之上扩展。

一种可行方向是：

1. Agent 编排器提交 token span 级语义 metadata：

```text
0-800: SYSTEM_PROMPT
800-950: USER_QUERY
950-1300: TOOL_OUTPUT
1300-1400: RESPONSE
```

2. LMCache 保存 KV chunk 时记录 semantic metadata。

3. Storage manager 和 eviction policy 消费 semantic metadata。

4. 驱逐策略从普通 LRU/LFU 扩展为语义优先级：

```text
SYSTEM_PROMPT      最后驱逐
PARTIAL_PREFILL    临时强保护
TOOL_OUTPUT        中等优先级
USER_QUERY         普通优先级
RESPONSE           优先驱逐
```

5. 如果要支持 partial prefill，还需要 vLLM connector 和 scheduler 配合。

这样做的好处是：LMCache 仍然保留外置 KV 管理和分层存储优势，同时吸收 Sutradhara 的语义感知能力。

### 6.1 结合后的直接好处

如果 LMCache 引入 Sutradhara 式语义 hints，收益主要来自三方面。

第一，**缓存保留更聪明**。当前 LMCache 可以复用精确 token 前缀，但不理解这段前缀为什么重要。加入语义 metadata 后，系统可以知道 `SYSTEM_PROMPT`、工具 schema、长期复用的工具输出比一次性 `RESPONSE` 更值得保留。这样在缓存空间紧张时，eviction policy 不再只看 LRU/LFU，而是优先淘汰低复用价值内容。

第二，**多轮 Agent 的命中率更稳定**。Agent 工作流里，系统提示词、工具 schema、历史上下文经常跨轮次复用；工具输出和最终回答则复用价值差异很大。语义 hints 可以帮助 LMCache 在多轮请求中保住高价值 KV chunk，减少“刚要复用就被淘汰”的情况。

第三，**外置 KV 管理仍然保持松耦合**。Sutradhara 的优势是语义强，但需要引擎和编排器深度协同；LMCache 的优势是外置 KV 管理、分层存储和跨 backend 复用。结合后的理想状态是：语义判断来自 orchestrator 或 connector，KV 存储、迁移、加载和隔离仍由 LMCache 负责。

### 6.2 对性能提升的预期

这种结合可能带来性能提升，但不能简单套用 Sutradhara 或 LMCache 论文中的数字。更准确的说法是：它会改善几个直接影响性能的指标。

| 指标 | 预期变化 | 原因 |
|---|---|---|
| Prefix / chunk 命中率 | 提升 | 高价值前缀更不容易被驱逐。 |
| TTFT / FTR | 降低 | 命中 KV 后减少 prefill 计算。 |
| Tail latency | 降低 | 多轮 Agent 中少一些高代价 cache miss。 |
| 有效吞吐 | 提升 | GPU 少做重复 prefill，可服务更多请求。 |
| Cache 污染 | 降低 | `RESPONSE`、一次性工具输出等低价值内容可少存或优先驱逐。 |

其中最有希望改善的是长系统提示词、多工具 schema、多轮工具调用这类场景。原因是这些 workload 里存在大量稳定前缀，如果它们被正确保留，LMCache 的 retrieve 能直接减少 prefill 成本。

但也有边界：

- 如果请求之间几乎没有共享前缀，语义 hints 也无法创造 KV 命中。
- 如果瓶颈主要在 decode，而不是 prefill，TTFT/FTR 可能改善，但端到端延迟改善有限。
- 如果只把 `phase` 放进 cache key，而不改 eviction policy，收益主要是隔离更准确，不会自动带来更高保留优先级。
- 如果要获得 Sutradhara 式 partial prefill 收益，还必须改 connector/scheduler；单靠 LMCache storage layer 不够。

因此，结合后的性能提升应通过 benchmark 验证，重点看 cache hit rate、TTFT/FTR、P90/P99 latency、GPU prefill 时间占比和 cache eviction 行为。可以参考 Sutradhara 用语义标签提升 hit rate 和降低 FTR 的思路，也可以参考 LMCache 论文中外置 KV 复用降低 TTFT、提升吞吐的评估方式，但最终数字需要在具体模型、负载和缓存容量下实测。[R4][R5][R9]

---

## 7. RFC：面向 UMBP 的 Semantic KV Cache 管理方案

把 Sutradhara、LMCache 和 Dynamo 的思路落到 MORI-UMBP 时，核心原则是：**不要让 UMBP core 理解 Agent 语义，而是在 UMBP 前面加一层 Semantic KV Adapter**。

UMBP 更像一个高性能 KV 仓库。它擅长把 KV block 放到 HBM/DRAM/SSD，不同节点之间查找和搬运 KV，并通过 master 给读写请求做路由建议。[M1][M2][M3][M4][M5][M6] 但它不应该判断某段 token 是系统提示词、用户问题、工具结果，还是最终回答。语义判断、key 生成和策略选择放在 adapter；UMBP 继续按 opaque key、tier 和轻量 metadata 执行存取。

本节只聚焦 adapter 的四个功能，避免把方案写成大而全的缓存系统。主线是 **semantic-driven cache policy optimization**：语义不是装饰字段，而是决定缓存价值、生命周期和准入策略的输入；同时，语义不能替代 key 正确性，真正的复用安全仍由 token chunk、模型、tokenizer、KV layout 和隔离字段保证。

这条语义优化链可以拆成四步：

1. **语义价值识别**：先把上游上下文映射成 `SYSTEM_PROMPT / TOOL_OUTPUT / PARTIAL_PREFILL / RESPONSE / UNKNOWN`，判断这段 KV 大致有什么缓存价值。
2. **语义生命周期策略**：紧接 phase，把语义阶段转成 TTL、priority 和临时 pin 策略。
3. **语义安全复用的 key 管理**：把 phase 与 token chunk、模型、tokenizer、KV layout、tenant/session 等一起编码进 canonical key，保证“值得缓存”不等于“可以误复用”。
4. **语义准入与 skip-save 控制**：根据 phase、hint、大小和命中统计决定哪些 KV 只做短期本地使用、哪些可以上报 external index、哪些值得交给 UMBP 托管。

### 7.1 职责边界

```text
Agent Orchestrator / SGLang / vLLM
        │
        │  原始上下文：
        │  - system message: 公司统一分析规范
        │  - user message:   分析华东区 Q2 销售异常
        │  - tool event:     SQL 查询结果
        │  - metadata:       tenant_id / agent_id / session_id
        ▼
UMBP Semantic KV Adapter
        │
        │  翻译结果：
        │  - system message -> semantic span + 高 priority hint
        │  - SQL result     -> phase=TOOL_OUTPUT，按大小/命中率决定是否保存
        │  - final answer   -> phase=RESPONSE，默认 skip-save
        │  - 输出给 UMBP: canonical key + metadata + put/report/skip 决策
        ▼
MORI-UMBP
        │
        │  执行动作：
        │  - metadata-only: match_external_kv() 找已有 KV 的节点
        │  - UMBP-owned:   RoutePut/RouteGet + batch_put/get 读写 KV bytes
        │  - policy hint:  使用 adapter 给出的 priority / ttl / tier hint 辅助分层和回收
```

这个边界有两层含义：

- **Adapter 负责翻译。** 它看 message role、tool event、session metadata 和 token span，把这些信息转成 semantic spans、标准化 key、保存策略和 UMBP API 调用。
- **UMBP 负责执行。** 它不解析 prompt，不判断语义，只按照 key、tier 和轻量 metadata 做路由、读写和回收。

UMBP master 和 peer 的关系可以理解为“路由建议”和“数据持有”的分工。master 通过 heartbeat 维护一个近似的 `GlobalBlockIndex`，为读写请求选择候选 peer；真正保存、读取和释放 KV block 的是 peer。[M2] 因此 adapter 上报给 master 的语义 metadata 应保持轻量，并尽量异步批量更新，不能让每个 token span 变化都变成同步 master 写入。

### 7.2 功能一：语义 phase 与缓存价值识别

第一个功能是把上层语义变成缓存策略可消费的价值信号。Adapter 不需要理解复杂业务，只需要把上游请求结构映射成少量稳定的 phase，用来区分“长期稳定前缀”“短期会话内容”“工具结果”“临时 prefill”和“默认不值得保存的 response”。phase 是后续 key、保存策略、TTL、priority 和 pin 的输入。

```text
SYSTEM_PROMPT     # 长期稳定前缀，默认高复用价值
USER_QUERY        # 当前请求或当前 session 的短期内容
TOOL_OUTPUT       # 工具结果，是否保存取决于大小、cacheable hint 和命中统计
PARTIAL_PREFILL   # 工具等待期间的临时关键前缀
RESPONSE          # 在线 decode 产生的最终回答，默认 skip-save
UNKNOWN           # 无法可靠判断的内容
```

phase 的来源可以分两类：

- **直接读取**：Chat API 的 `role=system/user/assistant/tool`、tool event、workflow metadata、`cacheable` 标记。
- **规则映射**：例如 `role=system` 映射为 `SYSTEM_PROMPT`，assistant response 默认映射为 `RESPONSE`，无法判断的 span 映射为 `UNKNOWN`。

这不是复杂业务理解，而是规则映射和默认值填充。phase 的作用也很明确：影响保存、上报、TTL、priority、pin，但不能替代 token chunk key。

功能一的直接输出不是最终 cache key，而是一组 semantic spans，例如：

```text
(token_start=0,    token_end=3072,  phase=SYSTEM_PROMPT, source=role:system)
(token_start=3072, token_end=3120,  phase=USER_QUERY,    source=role:user)
(token_start=3120, token_end=23500, phase=TOOL_OUTPUT,   source=tool_event)
```

这些 span 后续会在 key 管理阶段映射到推理引擎的 KV block / page chunk 上。功能一只负责给 token 区间打语义标签，不负责证明 KV bytes 可以复用；复用正确性仍由功能三的 chunk/key 管理保证。

### 7.3 功能二：语义生命周期策略

第二个功能是把语义价值进一步转成生命周期和调度 hint。它和功能一强绑定：phase 回答“这段 KV 是什么、价值大概在哪里”，priority / pin 回答“这段 KV 应该怎样被保存、保护和回收”。Adapter 可以生成：

```text
ttl_ms      # 多久后可以回收
priority    # 分层或驱逐排序时的优先级提示
pin_until   # 临时保护到某个事件完成
```

这些字段可以由编排器显式提供，也可以由 adapter 根据 phase 和配置给默认值。例如：

- `SYSTEM_PROMPT` 默认高 priority、较长 TTL。
- `USER_QUERY` 默认短 TTL。
- `TOOL_OUTPUT` 根据大小、工具类型和命中统计决定 priority。
- `PARTIAL_PREFILL` 可以 pin 到“工具调用返回”或“extend prefill 完成”。
- `RESPONSE` 默认 skip-save，因此通常不进入 pin/eviction 路径。

这里要区分两类依赖：`priority` 可以部分由 adapter 根据 phase 默认推导；`event-based pin` 更依赖上层 Agent / orchestrator 提供事件边界。没有显式事件 hint 时，adapter 不应该假设自己能判断业务生命周期，而应退化为 TTL-based 临时保护。

这些 hint 只是排序和保护信号。UMBP core 不需要知道“为什么这段内容重要”，只需要在路由、分层或回收策略中消费 adapter 给出的轻量 metadata。

### 7.4 功能三：语义安全复用的 key 管理

第三个功能是保证语义优化不会破坏复用正确性。phase 和 priority 只能说明“这段 KV 是否重要”，不能证明两次请求的 KV 可以复用。真正决定能否复用的，是 token chunk、模型、tokenizer 和 KV layout 是否一致。因此 adapter 要实现类似 LMCache 的 chunk/prefix key 管理。

Adapter 可以按推理引擎 connector 的 KV block / page 粒度切分 token chunk，并为每个 chunk 生成标准化 cache key：

```text
cache_key = hash(model_id, tokenizer_id, kv_layout_version,
                 token_chunk_hash, tenant_id, agent_id/workflow_id,
                 reuse_scope, phase)
```

这个标准化 key 也可以称为 canonical key，是 adapter 和 UMBP 之间的 KV 身份标识，作用有三种：

- **external KV report/match**：adapter 用 key 上报“这个 KV chunk 在哪个 worker、哪个 tier”；下一次请求生成同一个 key 后，用 `match_external_kv()` 找已有 KV 的节点。
- **UMBP-owned put/get**：adapter 用 key 调 `batch_put_from_ptr()` 保存 KV bytes；后续请求生成同一个 key 后，用 `batch_get_into_ptr()` 取回 KV bytes。
- **隔离和防错复用**：key 中包含 model、tokenizer、KV layout、tenant/session 和 phase，确保不同模型、不同租户或不同复用范围的 KV 不会误命中。

具体实现可以按下面顺序做：

```text
1. 从推理框架拿到 prompt tokens、model_id、tokenizer_id、kv_layout_version，以及功能一生成的 semantic spans。
2. 按推理引擎 connector 的 KV block / page 粒度切分 token chunk，并记录 token chunk 到 KV page 或 byte range 的映射。chunk 大小应来自引擎配置，不应写死。
3. 把 semantic span 落到 token chunk 上：
   - system message 覆盖的 chunk -> phase=SYSTEM_PROMPT
   - tool result 覆盖的 chunk     -> phase=TOOL_OUTPUT
   - assistant response 覆盖的 chunk -> phase=RESPONSE，默认 skip-save
4. 对每个 chunk 计算 token_chunk_hash，必要时保留 prefix hash 链，保证连续前缀可以逐块匹配。
5. 用 token_chunk_hash + model/tokenizer/layout + tenant/session/phase 生成标准化 key。
```

这里有三个关键点：

- **key 不能只包含 token hash。** 两个租户可能有相同系统提示词，但这不代表它们的 KV 可以互相复用。只有当 `reuse_scope = GLOBAL` 且明确授权时，才可以去掉 `tenant_id`。
- **语义边界和 chunk 边界不一定完全一致。** Adapter 应尽量在语义边界切 chunk；如果一个 chunk 同时包含 system prompt 末尾和 user query 开头，应使用更保守的 phase，例如 `UNKNOWN` 或较低 priority，避免把混合 chunk 当成长期 `SYSTEM_PROMPT` 保存。
- **`RESPONSE` 不是默认要保存的 prefill 前缀。** 在线 decode 产生的最终回答默认 skip-save；只有当它在下一轮对话中作为 history 重新进入 prompt 时，adapter 才按新的 prompt span 重新切分和标注。

### 7.5 功能四：语义准入与 skip-save 控制

第四个功能是把语义价值落实到缓存准入控制。功能三已经通过 canonical key 解决“能不能安全复用”；功能四回答另一个问题：即使一段 KV 可以被正确命名，它是否值得 report、put，或进入 UMBP-owned 路径。

Adapter 可以把每个 chunk 分成三类：

```text
skip        # 不 report、不 put，只让推理引擎本地按请求生命周期处理
report      # 只上报 metadata，用于后续 match_external_kv() 和路由
put         # 把 KV bytes 交给 UMBP 托管，用于跨节点、跨 tier 复用
```

准入策略应保守：

- `SYSTEM_PROMPT` 在 key 维度完整且 scope 合法时，默认至少 report；如果命中率高，可以进入 UMBP-owned `put` 路径。
- `USER_QUERY` 默认短 TTL，通常只在 session 内 report，不做长期 UMBP-owned 托管。
- `TOOL_OUTPUT` 需要结合大小、`cacheable` hint、工具类型和历史命中统计；信号不足时只 report 或直接 skip。
- `RESPONSE` 默认 skip-save，不 report、不 put；如果它下一轮作为对话历史重新进入 prompt，再按新的 prompt span 处理。
- `UNKNOWN` 或混合 chunk 默认 skip 或短 TTL report，避免污染 external KV index、HBM 和 DRAM。

这样功能四聚焦“是否进入缓存系统”，而不是重复功能三的 key 隔离。即使上游没有显式 hint，adapter 也可以用保守准入策略避免低价值 KV 占用热层，同时保留 metadata-only routing 带来的低风险收益。

### 7.6 Adapter 如何接入 UMBP

完成前四个功能后，adapter 已经有了 canonical key、policy hints 和 admission 决策。接下来并不是一定要把所有 KV bytes 都交给 UMBP；它可以选择两种接入深度：**只让 UMBP 知道 KV 在哪里**，或 **让 UMBP 真正保存 KV bytes**。

**方式一：metadata-only 接入，用 UMBP 做路由索引。**

KV bytes 仍然放在 SGLang/vLLM 自己的 cache 里。Adapter 只把 hash、tier、所在节点和少量语义 metadata 上报给 UMBP master，走 `report_external_kv_blocks()` / `match_external_kv()`。[M2][M5] 这时 UMBP 不搬运 KV bytes，只回答一个问题：哪些节点可能已经有这段 KV？router 可以据此把类似请求送到已有 KV 的节点，减少重复 prefill。

这条路径适合作为第一阶段：风险小，不改 UMBP 数据面，也不要求推理引擎马上把 KV layout 交给 UMBP 管。它的收益主要是路由层面的：

- Adapter 可以优先上报 `SYSTEM_PROMPT`、高命中 `TOOL_OUTPUT` 这类值得复用的 KV metadata，让后续请求更容易被路由到已有 KV 的节点。
- Adapter 可以不报告 `RESPONSE` 或低价值大对象，避免低复用内容污染 external KV index。
- `tenant_id / session_id / reuse_scope` 可以进入 key 或 metadata，减少跨租户、跨会话的错误匹配。
- `ttl_ms` 到期后，adapter 可以撤销或停止上报 metadata，让过期 KV 不再影响路由。

它不能带来完整的 UMBP tier placement 或驱逐收益，因为 KV bytes 仍由 SGLang/vLLM 自己管理；UMBP 只参与“找对节点”，不参与“保存和回收 bytes”。

**方式二：UMBP-owned 接入，让 UMBP 托管高价值 KV。**

当 adapter 判断某段 KV 值得长期复用，例如 `SYSTEM_PROMPT` 或高命中 `TOOL_OUTPUT`，它可以通过 `UMBPClient.batch_put_from_ptr()` / `batch_get_into_ptr()` 让 UMBP 直接保存和读取 KV bytes。[M2][M4] 这时 UMBP 会通过 `RoutePut` 选择 peer，peer 负责把 KV bytes 放到 HBM/DRAM/SSD；读取时再通过 `RouteGet` 从最快 tier 取回。

这条路径收益更完整：

- 高价值 KV 不再只依赖某个 vLLM/SGLang worker 的本地 cache，而是可以由 UMBP 统一托管。
- Adapter 可以把 `SYSTEM_PROMPT` 这类高复用 KV 标成高 priority / 长 TTL；UMBP 在现有分层和后续策略扩展中消费这些 hint，优先保留在 HBM/DRAM，必要时用 SSD 做兜底副本。
- 后续请求即使命中在不同节点，也可以通过 `RouteGet` 和 `batch_get_into_ptr()` 把缺失 KV chunk 取回，减少重复 prefill。
- 当 HBM/DRAM 有压力时，adapter 提供的 `phase / priority / ttl` 可以参与驱逐排序，让 `SYSTEM_PROMPT` 比低命中 `TOOL_OUTPUT` 更晚被回收。

代价是 adapter 必须正确处理 page layout、dtype、model id、tokenizer id 和 KV layout version，否则不同模型或不同 KV layout 的 bytes 可能被错误复用。因此这条路径更适合作为第二阶段，在 metadata-only routing 验证收益后再接入。

因此，推荐落地顺序是：adapter 先生成 semantic spans、policy hints 和 canonical key，并用于 metadata-only routing；确认路由收益后，再把少量高价值 KV 接入 UMBP-owned 路径；最后再让 UMBP core 更深入地消费 priority、TTL 和 pin hint。这样每一步都能独立验证，不需要一次性改 UMBP core。

### 7.7 端到端例子

假设一个企业报表 Agent 要分析“华东区 Q2 销售异常”。一次请求大致包含：

```text
SYSTEM_PROMPT:  公司统一分析规范，约 3K tokens
USER_QUERY:     “分析华东区 Q2 销售异常”
TOOL_OUTPUT:    SQL 查询结果，约 20K tokens
PARTIAL_PREFILL: 等待 SQL 时已 prefill 的工具无关上下文
RESPONSE:       最终中文报告
```

**第一层：Agent / 推理框架提供上下文。**

上游不需要懂 UMBP，只要保留请求结构即可：system message 是公司统一规范，user message 是本次问题，tool event 是 SQL 查询结果，metadata 里有 `tenant_id / agent_id / session_id / model_id`。这些信息本来就存在于 Agent runtime 或推理框架 adapter 中，区别只是以前没有传给 KV cache 策略使用。这一层只提供事实，不决定 KV 该不该保存。

**第二层：Semantic KV Adapter 把每段上下文翻译成缓存决策。**

- 对公司统一分析规范这 3K tokens，adapter 识别为 `SYSTEM_PROMPT`。在 chunk 边界、model、tokenizer、KV layout、tenant/reuse scope 都匹配的前提下，它用 `tenant_id + agent_id + model_id + token_hash + phase` 生成稳定 key，并把策略设为“高优先级保存、优先 HBM/DRAM、必要时 SSD 兜底”。原因是同一个企业报表 Agent 后续很多请求都会复用这段规范。
- 对“分析华东区 Q2 销售异常”这句用户问题，adapter 识别为 `USER_QUERY`。它把 key 绑定到 `session_id`，设置短 TTL。原因是这段内容通常只对当前会话的后续几轮有用，不应该变成跨会话长期缓存。
- 对 20K tokens 的 SQL 查询结果，adapter 识别为 `TOOL_OUTPUT`。它不会凭空知道这个 SQL 是否“常用”，而是看上游是否带了 `tool_name / workflow_id / cacheable` 之类的 hint，或者看配置规则和历史命中统计。信号足够时可以 report 或 put；信号不足时就降低优先级或 skip。原因是大工具结果很容易挤占 HBM/DRAM。
- 对等待 SQL 时已经 prefill 的工具无关上下文，adapter 识别为 `PARTIAL_PREFILL`，在工具返回前临时 pin。原因是这部分如果被驱逐，工具返回后还要重复 prefill。
- 对在线 decode 产生的最终中文报告，adapter 识别为 `RESPONSE`，默认 skip-save。原因是最终回答很少作为后续请求的稳定前缀；如果它下一轮作为对话历史重新进入 prompt，再按新的 prompt span 处理。

**第三层：UMBP 按这些具体决策执行。**

到 UMBP 这一层时，业务语义已经被 adapter 翻译成 key、API 调用、tier policy 和轻量 metadata。UMBP 不再判断“报表规范是否重要”，只按这些输入执行：

1. **系统提示词先用于路由。** 如果某个 worker 已经有“公司统一分析规范”的 KV，adapter 上报这段 `SYSTEM_PROMPT` 的 hash、节点和 tier。下一次同一 tenant 的报表请求进来时，adapter 调 `match_external_kv()`；UMBP master 返回可能命中的节点，router 优先把请求送过去，减少重复 prefill。
2. **高价值内容再写入 UMBP。** 对 `SYSTEM_PROMPT`，adapter 可以按默认规则认为它复用价值高；对 `TOOL_OUTPUT`，adapter 需要依据上游 hint、配置规则或历史命中统计做 admission 判断。确认值得托管后，adapter 调用 `batch_put_from_ptr()`。UMBP master 通过 `RoutePut` 选择 peer；peer 在 HBM/DRAM/SSD 中分配空间，真正保存 KV bytes。
3. **后续请求按 key 取回缺失 chunk。** 当新请求只缺少部分 KV chunk 时，adapter 调 `batch_get_into_ptr()`。UMBP master 用 `RouteGet` 查 `GlobalBlockIndex`，优先从 HBM/DRAM 返回；如果热层没有但 SSD 有副本，再从 SSD 回填。
4. **缓存压力下按 hint 排序。** HBM/DRAM 空间紧张时，UMBP 不理解“报表规范”这个业务含义，但可以使用 adapter 给出的 `phase / priority / ttl` 作为排序信号：`SYSTEM_PROMPT` 尽量保留，低命中 `TOOL_OUTPUT` 下沉或驱逐，`RESPONSE` 因为 skip-save 不占 cache。

这个例子的重点不是让 UMBP 理解“报表”或“SQL”的业务含义，而是让 adapter 把这些业务上下文变成正确 key 和可执行策略：系统提示词稳定复用，所以尽量保留；用户问题只在会话内短期保留；工具结果要看是否常用；最终回答默认不保存。

这样得到的系统形态可以概括为：Sutradhara 式 hint 提供语义 phase 和事件边界，LMCache 式 key 管理提供 cache identity 与隔离规则，UMBP 负责按 opaque key 和轻量 metadata 做路由、存储与跨 tier 搬运。

### 7.8 小结：Adapter 的职责和收益

Semantic KV Adapter 的核心职责不是替代 UMBP，而是把 Agent/推理框架里的上下文翻译成 UMBP 可以安全消费的 key、metadata 和 API 调用。围绕 semantic-driven cache policy optimization，adapter 需要做这些事：

- **识别语义价值**：优先读取上游已有字段，例如 message role、tool event、workflow metadata 和 `cacheable` 标记；缺失时按简单规则补默认值，例如 `role=system` 对应 `SYSTEM_PROMPT`。
- **生成语义生命周期策略**：基于 phase 生成 priority 和 TTL；对 event-based pin，则优先消费上层 Agent / orchestrator 提供的事件边界，缺失时退化为 TTL-based 临时保护。
- **保证语义安全复用**：按 token chunk、model、tokenizer、KV layout、tenant/session、reuse scope 和 phase 生成标准化 cache key，让语义价值指导策略，但不替代复用正确性判断。
- **执行语义准入控制**：根据 phase、priority、TTL、大小和命中统计决定 skip、report 还是 put，避免低价值 KV 污染 external KV index 和热层缓存。

预期收益也应分阶段看。metadata-only 阶段主要提升 routing 质量：请求更容易被送到已有 KV 的节点，减少重复 prefill，同时避免低价值 KV 污染 external KV index。UMBP-owned 阶段进一步带来跨节点 KV 复用和分层存储收益：高价值 KV 可以由 UMBP 托管，从 HBM/DRAM/SSD 中按最快可用 tier 取回。后续如果 UMBP 消费 adapter 提供的 `phase / priority / ttl`，还能减少 cache 污染，让系统提示词这类高复用 KV 比低命中工具结果更晚被回收。

---

## 8. 最终结论

Sutradhara、LMCache、Dynamo 代表了 Agent Hints 在 KV Cache 系统中的三种路线。

**Sutradhara** 代表语义最强的路线。它让编排器直接参与引擎缓存管理，适合追求极致 Agent 工作流优化，但工程耦合高。

**LMCache** 代表缓存管理基础设施路线。它当前没有原生 `agent_hints`，但能把 KV Cache 作为外部资源进行查询、固定、迁移、清理、压缩和分层存储，是构建更高级 Agent-aware KV 系统的基础。

**Dynamo** 代表分布式路由优化路线。它通过请求级 hints 提升多 Worker 场景下的 KV 局部性和调度质量，特别适合生产集群。

长期来看，更理想的方向是把这些能力组合起来：

- Dynamo 做请求路由。
- LMCache 做外置 KV 管理和分层存储。
- Sutradhara 式语义 metadata 指导缓存优先级和生命周期。
- 对 UMBP 来说，合理方向是让语义 hints 进入 adapter 和轻量 metadata，而不是破坏其 master-as-advisor、peer-owned 数据面的基础设计。

---

## 9. 参考资料与代码出处

### 外部资料

- [R1] 用户提供的调研报告 `report.md`：第 2.1 节 “Sutradhara：Orchestrator-Engine Co-design 与语义标记”；第 3.1、3.4 节对语义标记和复用优先级的总结。报告引用 Biswas et al., “Sutradhara: An Intelligent Orchestrator-Engine Co-design for Tool-based Agentic Inference.”
- [R2] 用户提供的调研报告 `report.md`：第 2.2 节 “NVIDIA Dynamo：Agent Hints 与 KV 感知路由”；第 3.2、3.3 节对路由提示和生命周期控制的总结。
- [R3] 用户提供的调研报告 `report.md`：第 4.1 节 “Co-design vs. Layered Architecture”；第 4.2 节对 LMCache 作为引擎无关连接器方案的总结。
- [R4] Sutradhara arXiv HTML：`https://arxiv.org/html/2601.12967`。该页面列出 `submit_partial_prefill()`、`extend_prefill()`、`register_streaming_callback()`、`tag_kv_blocks()`、`set_reuse_priority()` 等 API，并描述 partial prefill、语义 KV block tagging 和 priority-based eviction。
- [R5] Microsoft Research 发布页：`https://www.microsoft.com/en-us/research/publication/sutradhara-an-intelligent-orchestrator-engine-co-design-for-tool-based-agentic-inference/`。该页总结了 Sutradhara 的 co-design 动机、三类优化，以及在 vLLM/A100 上降低 FTR 和端到端延迟的结果。
- [R6] NVIDIA Dynamo 官方 Agents 文档：`https://docs.dynamo.nvidia.com/dynamo/user-guides/agents`。该页说明 agent-facing request metadata 位于 `nvext`，`agent_hints` 可携带 priority、expected output length、speculative prefill 等 serving-relevant intent，并提到 priority、osl、planned `context_type` 等字段。
- [R7] NVIDIA NeMo Agent Toolkit Dynamo LLM provider 源码/文档：`https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/packages/nvidia_nat_core/src/nat/llm/dynamo_llm.py` 与 `https://docs.nvidia.com/nemo/agent-toolkit/latest/api/nat/llm/dynamo_llm/index.html`。该实现说明所有 routing hints 注入到 `nvext.agent_hints`，标准字段包括 `latency_sensitivity`、`osl`、`priority`，自定义字段包括 `prefix_id`、`total_requests`、`iat`，并注入 `nvext.cache_control`。
- [R8] Dynamo Router README：`https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/README.md`。该文档说明 KV Router 通过 KV cache overlap、prefill/decode cost 进行路由，并提到 `nvext.agent_hints.priority` 与 router queue priority scheduling。
- [R9] LMCache arXiv HTML：`https://arxiv.org/html/2510.09665`。该论文 Figure 8 对比 LMCache、basic vLLM、basic vLLM CPU offloading 和商业方案的 TTFT 与 throughput，并在摘要和评估部分总结最高 15x 吞吐量提升、至少 2x 延迟降低等结果。

### LMCache 代码出处

- [C1] `lmcache/integration/vllm/vllm_v1_adapter.py`：`extract_request_configs()` 从 `kv_transfer_params` 中提取 `lmcache.` 前缀字段；请求 tracker 保存 `request_configs`；adapter 在保存路径检查 `lmcache.skip_save`。
- [C2] `lmcache/utils.py`：`CacheEngineKey` 从 `request_configs` 中提取 `lmcache.tag.*` 到 `tags`，并在 `__hash__`、`__eq__`、`to_string()` 中使用这些 tag。
- [C3] `lmcache/v1/token_database.py`：`ChunkedTokenDatabase` 按 `chunk_size` 切分 token、生成 prefix hash，并通过 `_make_key_by_hash()` 构造 `CacheEngineKey`。
- [C4] `lmcache/v1/cache_engine.py`：`lookup()`、`move()`、`compress()`、`decompress()`、`clear()`；`lookup_pins` 和 `lookup_unpin()` 管理 lookup 命中后的 pin/unpin。
- [C5] `lmcache/integration/vllm/lmcache_mp_connector.py` 与 `lmcache/integration/vllm/vllm_multi_process_adapter.py`：MP connector 的 lookup、store、retrieve metadata 构造；按 LMCache 命中 token 数决定需要外部加载的 token 范围；worker 提交 store/retrieve 请求。
- [C6] `lmcache/v1/multiprocess/custom_types.py`：`IPCCacheServerKey` 中 `request_id` 是 session tracking，不属于缓存身份；`cache_salt` 是 per-user isolation salt，属于缓存身份。
- [C7] `lmcache/v1/distributed/l2_adapters/base.py`、`lmcache/v1/distributed/eviction_policy/isolated_lru.py`、`lmcache/v1/multiprocess/http_apis/quota_api.py`：分布式 L2 使用量、quota 和 isolated eviction 按 `cache_salt` 维度管理。
- [C8] `lmcache/v1/config.py`、`lmcache/v1/storage_backend/__init__.py`、`lmcache/v1/storage_backend/connector/__init__.py`：本地 CPU、本地磁盘、远程 backend、Redis/RESP 等分层和远程存储配置入口。

### MORI / UMBP 代码出处

- [M1] `mori/README.md`：MORI 是 RDMA + GPU 通信框架，UMBP 是 unified memory & bandwidth pool，提供 tiered storage 和 distributed key-value access。
- [M2] `mori/src/umbp/doc/design-master-control-plane.md`：UMBP master-as-advisor、peer-owned allocator、`GlobalBlockIndex`、`ExternalKvBlockIndex`、`RouteGet` / `RoutePut`、HBM/DRAM/SSD tier、eviction 和 hot-path Put/Get 流程。
- [M3] `mori/src/umbp/doc/runtime-env-vars.md`：UMBP runtime knobs，包括 heartbeat、lease、SSD、SPDK proxy tenant quota、`UMBP_CACHE_REMOTE_FETCHES` 等。
- [M4] `mori/tests/python/umbp/test_umbp_client_ptr.py`：Python `UMBPClient` 指针 API 示例，包括 `put_from_ptr`、`get_into_ptr`、`batch_put_from_ptr`、`batch_get_into_ptr` 和 SSD-enabled round trip。
- [M5] `mori/tests/python/umbp/test_umbp_master_client.py`：External KV block report/match/revoke、multi-tier registration、hit count 和 master client 行为测试。
- [M6] `mori/src/umbp/include/umbp/umbp_client.h`、`mori/src/umbp/local/tiers/local_storage_manager.cpp`、`mori/tests/cpp/umbp/local/test_prefix_aware_eviction.cpp`：`BatchPutWithDepth` 和 `prefix_aware_lru` 使用 radix-tree chain depth 作为本地 eviction priority 信号。
