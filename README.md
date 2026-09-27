# shard-router

**长上下文自动切片 + 多 key 并发 + 结果汇总的 OpenAI 兼容网关。**

解决一个具体问题：某个 provider（如 SenseNova kimi-k3）按 **TPM（每分钟 token 数）** 限流，
prompt 一大就撞 `EndpointTPMExceeded`。本服务把一个超长请求自动切成若干小片，
每片分配一个不同的 API key **并发**下发，最后用一次 reduce 调用把各片结论汇总成最终答案。

效果：
- **不再因 TPM 撞墙**：每片都远小于单 key 的分钟 token 上限。
- **总时长更短**：N 片并发，耗时 ≈ max(各片) + reduce，而非 sum(各片)。
- **小请求零损耗**：低于阈值的请求原样直连，行为与普通网关一致。

## 架构

```
客户端 ──► shard-router (:20200)
              │
              ├─ est_tokens < threshold ──► 直接透传单 key（不切片）
              │
              └─ est_tokens ≥ threshold ──► split_messages() 切成 N 片
                                              │
                                              ▼
                              N 个 key 并发下发（KeyPool 轮询+冷却）
                                              │
                                              ▼
                                      收集 N 份结论
                                              │
                                    ┌─────────┴─────────┐
                              reduce_enabled=true   false
                                    │                  │
                            再调一次模型汇总      直接拼接返回
```

## 快速开始

```bash
cd /opt/shard-router
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 方式一：环境变量提供 keys（推荐，避免落盘）
export SENSENOVA_KEYS="sk-xxx,sk-yyy,sk-zzz,..."
.venv/bin/python server.py

# 方式二：直接写在 config.yaml 的 providers.sensenova.keys 里
```

打开 http://127.0.0.1:20200 可看状态页、改切片策略、测连通。

## 接入 9router

新增一个 OpenAI 兼容 provider：

| 字段 | 值 |
|---|---|
| base_url | `http://172.17.0.1:20200/v1` |
| api key | 任意非空字符串（key 由本服务管） |
| 模型 | `kimi-k3` / `deepseek-flash` / `glm-5.2` |

然后把该 provider 的模型加进你要用的 combo。

## 切片策略说明

配置在 `config.yaml` 的 `sharding` 段：

| 参数 | 含义 |
|---|---|
| `token_threshold` | 估算 token 超过此值才切片。低于则透传。 |
| `shard_tokens` | 每片的目标 token 数。设成单 key TPM 上限的 50%~70%。 |
| `max_shards` | 最大并发片数，通常等于 key 数。 |
| `replicate_system` | 每片是否都带 system prompt。对话场景必须 true。 |

**切分规则**（`app/shard.py`）：

1. `system` 消息按 `replicate_system` 决定是否复制到每片头部。
2. 其余消息按累积 token 贪心分组，满 `shard_tokens` 开新片。
3. **单条消息本身超过 `shard_tokens`** 时，按段落→句子→字符三级边界切碎，
   每段加 `（此为用户长消息的第 N/M 段）` 标记。
   这是关键：TPM 撞墙的根源正是一条塞满几十万 token 的请求。
4. 非纯文本的超大消息（巨型 `tool_calls` / 多模态 parts）不切内容，单独成片。
5. 分片数超 `max_shards` 时，尾部合并到上限内。

### 一个重要取舍：不丢内容 > 不超预算

切分过程中任何一步都**不裁掉正文**。宁可单片略微超出 `shard_tokens`，
也绝不静默截断——模型看漏内容比多花几十个 token 严重得多。
`tests/` 里 `test_giant_message_split_anywhere_loses_no_text` 专门锁死这条性质。

## 多 key 池

`app/pool.py`：

- 轮询分配，每次请求天然限制并发数在 `max_shards` 内——这同时对 RPM 限流形成节流。
- 单 key 失败（429/5xx/超时）后**指数退避冷却**：30s → 60s → 120s → … 封顶 300s。
- 全冷却时短暂等待而非直接报错。
- `Accept-Encoding: identity`：防 gzip 攒缓冲拖慢 TTFT。

## 汇总（reduce）

分片各自只看到一部分上下文，直接拼给客户端会让模型「失忆」。
`reduce_enabled=true` 时用一次额外请求，把 N 份结论合成连贯答案。
reduce 用 `aggregation.reduce_model`（默认同分片模型），也走 key 池。

reduce 失败不致命：自动退回「直接拼接」结果。

## API

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 兼容，支持 `stream: true`（分片为一次性 chunk 流） |
| `GET /api/status` | provider/池子/配置/统计数据 |
| `POST /api/config` | 全量保存配置并热加载（无需重启） |
| `POST /api/test` | 发一次探测请求 |
| `GET /healthz` | 健康检查 |
| `GET /` | Web UI |

响应带 `x_shard_router` 字段，记录本次是否切片、每片 token、每片耗时和用的 key，
便于排查：

```json
"x_shard_router": {
  "sharded": true, "reason": "over_threshold",
  "est_tokens": 25796, "shard_count": 9,
  "reduce_used": true, "latency_s": 0.23,
  "per_shard": [{"index": 0, "ok": true, "tokens": 6035, "latency_s": 0.11, "key": "AAA111"}]
}
```

## 测试

```bash
.venv/bin/python -m pytest tests/ -v
```

40 个测试，覆盖：token 估算、切片边界、不丢正文性质、system 复制、
key 冷却与轮询、并发性（4 片各 sleep 0.3s 总耗时须 < 1s）、reduce 开关、
reduce 失败降级、全失败报错、OpenAI 响应结构。

## 已知限制

1. **分片可能不均匀**：贪心分组下，大片旁的小消息会形成小片。
   不影响正确性（每片都在预算内、所有 key 都会用上），但 TPM 分布不均衡。
2. **流式是伪流式**：分片内部并发放完才一次性 chunk 返回。
   真流式需要放弃「全部到齐再 reduce」。
3. **token 是估算**：没有 tiktoken，用 CJK≈1 token/字、其他≈4 token/字符。
   偏保守（宁多算），对「是否切片」的决策够用。
4. **对话历史切分会丢关联**：分片只看部分历史，靠 reduce 兜底。
   若任务是强依赖的顺序推理（如精读后逐段改写），本方案不适用——
   那种场景该用「滚动摘要」而不是分片。

## 文件

| 路径 | 作用 |
|---|---|
| `server.py` | FastAPI 入口，OpenAI 兼容端点 + 管理 API |
| `app/shard.py` | token 估算 + 切片算法 |
| `app/pool.py` | 多 key 池，轮询 + 指数冷却 |
| `app/router.py` | 路由决策 + 并发下发 + reduce 汇总 |
| `app/config.py` | YAML 配置加载与热加载 |
| `static/index.html` | Web UI（状态/策略/测试） |
| `tests/` | 40 个单元测试 |
| `config.yaml` | 主配置 |
