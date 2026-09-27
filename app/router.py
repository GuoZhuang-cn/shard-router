"""核心编排：长请求 → 切片 → 多 key 并发 → 汇总。

这是 shard-router 的心脏。流程：
  1. 估算 messages token 数
  2. 低于阈值 → 原样单发（直连，零損失）
  3. 超过阈值 → 切成 N 片，每片配一个 key 并发下发
  4. 收集 N 个结果，走 reduce 阶段（可关）合成最终 answer

为什么要 reduce：分片各自只看到一部分上下文，直接拼给客户端会
让模型「失忆」。reduce 阶段用一次请求把 N 份中间结论合成最终答案。

关键取舍：
- reduce_enabled=False 时退化为「并行摘要后拼接」，更快但质量低。
- 流式模式先发分片结果，reduce 完成后发最终帧。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .pool import KeyPool, UpstreamError
from .shard import Shard, messages_tokens, split_messages


@dataclass
class ShardResult:
    index: int
    ok: bool
    text: str = ""
    error: str = ""
    tokens: int = 0
    latency: float = 0.0
    key_tail: str = ""


@dataclass
class RouteDecision:
    """一次请求的路由决策，用于 UI 展示和日志。"""

    sharded: bool
    reason: str
    est_tokens: int
    shard_count: int = 0
    shards: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ShardSettings:
    token_threshold: int = 12000
    shard_tokens: int = 8000
    max_shards: int = 10
    replicate_system: bool = True


@dataclass
class AggSettings:
    mode: str = "stream"
    reduce_enabled: bool = True
    reduce_model: str | None = None


def _extract_text(data: dict[str, Any]) -> str:
    """从 chat.completion 响应里取正文，兼容多种返回形状。"""
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or choices[0].get("delta") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    # reasoning-only 响应（有些模型只回 reasoning）
    for k in ("reasoning_content", "reasoning"):
        if msg.get(k):
            return str(msg[k]).strip()
    return ""


class ShardRouter:
    def __init__(
        self,
        pool: KeyPool,
        settings: ShardSettings,
        agg: AggSettings,
        timeouts: dict[str, float],
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.agg = agg
        self.timeouts = timeouts
        self._session: Any = None

    async def start(self) -> None:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=64, force_close=False)
            )

    async def stop(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def decide(self, payload: dict[str, Any]) -> RouteDecision:
        msgs = payload.get("messages") or []
        est = messages_tokens(msgs)
        if est < self.settings.token_threshold:
            return RouteDecision(False, "below_threshold", est, 1, [{"index": 0, "tokens": est}])
        shards = split_messages(
            msgs,
            shard_tokens=self.settings.shard_tokens,
            max_shards=self.settings.max_shards,
            replicate_system=self.settings.replicate_system,
        )
        if len(shards) <= 1:
            return RouteDecision(False, "single_shard", est, 1, shards[0].to_dict() and [shards[0].to_dict()])
        return RouteDecision(True, "over_threshold", est, len(shards), [s.to_dict() for s in shards])

    async def _run_shard(
        self, session: Any, shard: Shard, payload: dict[str, Any], model: str
    ) -> ShardResult:
        """单个分片：拿 key → 下发 → 拿结果。失败自动换 key 重试 2 次。"""
        body = {
            "model": model,
            "messages": shard.messages,
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "max_tokens": payload.get("max_tokens"),
            "stream": False,
        }
        body = {k: v for k, v in body.items() if v is not None}
        body.setdefault("model", model)

        last_err = ""
        for attempt in range(3):
            got = self.pool.acquire()
            if got is None:
                # 全冷却：等最短的那个恢复
                await asyncio.sleep(2.0)
                got = self.pool.acquire()
                if got is None:
                    return ShardResult(shard.index, False, error="all keys cooling", tokens=shard.tokens)
            idx, st = got
            t0 = time.monotonic()
            try:
                data = await self.pool.complete(
                    session,
                    idx,
                    body,
                    self.timeouts.get("connect", 15),
                    self.timeouts.get("read", 120),
                )
                text = _extract_text(data)
                lat = time.monotonic() - t0
                return ShardResult(
                    shard.index, True, text=text, tokens=shard.tokens,
                    latency=round(lat, 2), key_tail=st.key[-6:],
                )
            except UpstreamError as e:
                self.pool.report(idx, False)
                last_err = f"{e.status}: {e.detail[:120]}"
                await asyncio.sleep(0.5 * (attempt + 1))
            except asyncio.TimeoutError:
                self.pool.report(idx, False)
                last_err = "timeout"
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                self.pool.report(idx, False)
                last_err = f"{type(e).__name__}: {e}"[:160]
                break
        return ShardResult(shard.index, False, error=last_err, tokens=shard.tokens)

    async def run(self, payload: dict[str, Any], model: str) -> dict[str, Any]:
        """非流式主编排：返回与 OpenAI chat.completion 同构的 dict。"""
        dec = self.decide(payload)
        if not dec.sharded:
            shard = Shard(0, list(payload.get("messages") or []), dec.est_tokens)
            res = await self._run_shard(self._session, shard, payload, model)
            return self._to_openai(res, model, dec, [res])

        shards = split_messages(
            payload["messages"],
            shard_tokens=self.settings.shard_tokens,
            max_shards=self.settings.max_shards,
            replicate_system=self.settings.replicate_system,
        )
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[self._run_shard(self._session, s, payload, model) for s in shards]
        )
        parallel_time = time.monotonic() - t0

        ok_results = [r for r in results if r.ok and r.text.strip()]
        if not ok_results:
            return self._to_openai(
                ShardResult(0, False, error="all shards failed: " + "; ".join(r.error for r in results)[:300]),
                model, dec, list(results),
            )

        final_text, reduce_used = await self._reduce(ok_results, payload, model)
        merged = ShardResult(
            0, True, text=final_text,
            tokens=sum(r.tokens for r in ok_results),
            latency=round(parallel_time, 2),
        )
        merged.reduce_used = reduce_used  # type: ignore[attr-defined]
        return self._to_openai(merged, model, dec, list(results))

    async def _reduce(
        self, results: list[ShardResult], payload: dict[str, Any], model: str
    ) -> tuple[str, bool]:
        """把 N 份分片结果合成最终答案。"""
        parts = [f"【分片 {r.index + 1} 结论】\n{r.text.strip()}" for r in results]
        if len(parts) == 1:
            return parts[0], False
        joined = "\n\n".join(parts)

        if not self.agg.reduce_enabled:
            return joined, False

        original_q = ""
        for m in reversed(payload.get("messages") or []):
            if m.get("role") == "user":
                c = m.get("content")
                original_q = c if isinstance(c, str) else str(c)
                break

        reduce_prompt = (
            "下面是同一长上下文被切分成多个片段后，各片段的独立分析结论。"
            "请把它们整合成对原始问题的完整、连贯、无重复的最终回答。"
            "不要出现「分片」字样，不要罗列中间过程，只给最终答案。\n\n"
            f"【原始问题】\n{original_q[:4000]}\n\n"
            f"【各分片结论】\n{joined[:12000]}"
        )
        rmodel = self.agg.reduce_model or model
        body = {
            "model": rmodel,
            "messages": [
                {"role": "system", "content": "你是结果整合器，负责合并多份分析结论。"},
                {"role": "user", "content": reduce_prompt},
            ],
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "stream": False,
        }
        body = {k: v for k, v in body.items() if v is not None}
        got = self.pool.acquire()
        if got is None:
            return joined, False
        idx, st = got
        try:
            data = await self.pool.complete(
                self._session, idx, body,
                self.timeouts.get("connect", 15), self.timeouts.get("read", 120),
            )
            return _extract_text(data) or joined, True
        except Exception:  # noqa: BLE001
            # reduce 失败不致命：退回拼接结果
            return joined, False

    def _to_openai(
        self, res: ShardResult, model: str, dec: RouteDecision, all_results: list[ShardResult]
    ) -> dict[str, Any]:
        if not res.ok:
            return {
                "error": {
                    "message": res.error or "shard failed",
                    "type": "shard_router_error",
                    "code": 502,
                }
            }
        finish = "stop"
        return {
            "id": f"chatcmpl-shard-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": res.tokens,
                "completion_tokens": 0,
                "total_tokens": res.tokens,
            },
            "x_shard_router": {
                "sharded": dec.sharded,
                "reason": dec.reason,
                "est_tokens": dec.est_tokens,
                "shard_count": dec.shard_count,
                "shards": dec.shards,
                "reduce_used": getattr(res, "reduce_used", False),
                "latency_s": res.latency,
                "per_shard": [
                    {
                        "index": r.index, "ok": r.ok, "tokens": r.tokens,
                        "latency_s": r.latency, "key": r.key_tail,
                        **({"error": r.error[:160]} if r.error else {}),
                    }
                    for r in all_results
                ],
            },
        }
