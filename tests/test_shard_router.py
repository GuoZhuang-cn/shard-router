"""shard-router 单元测试：重点验证切片正确性和并发下发逻辑。

跑法：.venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.pool import KeyPool, ProviderConfig, UpstreamError  # noqa: E402
from app.router import AggSettings, ShardRouter, ShardSettings, _extract_text  # noqa: E402
from app.shard import Shard, estimate_tokens, messages_tokens, split_messages  # noqa: E402


# ---------- token 估算 ----------

def test_estimate_tokens_chinese_denser_than_english():
    zh = estimate_tokens("你好世界")
    en = estimate_tokens("hello world")
    # 4 个汉字 ≈ 4 token；11 个英文字符 ≈ 3 token
    assert zh >= en


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_messages_tokens_counts_structure_overhead():
    n = messages_tokens([{"role": "user", "content": "abc"}])
    assert n >= 4  # 至少包含结构开销


def test_messages_tokens_handles_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "你好"}, {"type": "image_url"}]}]
    assert messages_tokens(msgs) > 512


# ---------- 切片 ----------

def _mk(turns=10, size=400):
    """造 size 字符/条的历史对话。"""
    msgs = [{"role": "system", "content": "你是助手"}]
    for i in range(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": "x" * size + f"#{i}"})
    return msgs


def test_no_split_below_threshold():
    msgs = _mk(turns=2, size=100)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10)
    assert len(shards) == 1


def test_split_produces_multiple_shards():
    msgs = _mk(turns=30, size=3000)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10)
    assert len(shards) > 1


def test_every_shard_carries_system_when_replicate_true():
    msgs = _mk(turns=30, size=3000)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10, replicate_system=True)
    for s in shards:
        assert s.messages[0]["role"] == "system"
        assert s.messages[0]["content"] == "你是助手"


def test_system_not_replicated_when_false():
    msgs = _mk(turns=30, size=3000)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10, replicate_system=False)
    for s in shards:
        assert all(m["role"] != "system" for m in s.messages)


def test_shard_count_capped_at_max():
    msgs = _mk(turns=100, size=5000)
    shards = split_messages(msgs, shard_tokens=1000, max_shards=5)
    assert len(shards) <= 5


def test_oversized_text_message_gets_content_split():
    """纯文本巨型消息现在按内容切（不再是单独成片）。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a" * 50000},
    ]
    shards = split_messages(msgs, shard_tokens=8000, max_shards=20, replicate_system=False)
    assert len(shards) > 1
    for s in shards:
        assert s.tokens <= 8000 * 1.4, f"分片超限: {s.tokens}"


def test_no_message_lost_across_shards():
    """所有非 system 消息必须被完整切分，一条不丢。"""
    msgs = _mk(turns=50, size=1500)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10, replicate_system=False)
    flat = [m for s in shards for m in s.messages]
    original = [m for m in msgs if m["role"] != "system"]
    assert len(flat) == len(original)
    # 顺序也保持
    assert [m["content"] for m in flat] == [m["content"] for m in original]


def test_shard_tokens_are_estimated():
    msgs = _mk(turns=30, size=3000)
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10)
    for s in shards:
        assert s.tokens == messages_tokens(s.messages)


def test_giant_message_is_content_split_not_isolated():
    """核心场景：一条 5 万字符的巨型消息必须被按内容切开，而不是独占一片。

    这正是 TPM 撞墙的根源——单条塞满几十万 token 的请求。
    """
    giant = "段落内容。" * 8000  # ~4 万字符
    msgs = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": giant},
        {"role": "user", "content": "请总结"},
    ]
    shards = split_messages(msgs, shard_tokens=8000, max_shards=20, replicate_system=False)
    assert len(shards) > 1, "巨型消息应该被切成多片"
    # 每片都必须在预算内（这是避免 TPM 撞墙的关键）
    for s in shards:
        assert s.tokens <= 8000 * 1.35, f"分片超限: {s.tokens} tokens"


def test_giant_message_split_anywhere_loses_no_text():
    """切段采用段落/句子边界，仅允许截断换行，不允许丢正文。"""
    giant = "第%d段内容。" % 0
    giant = "".join(f"第{i}段内容。" for i in range(2000))
    msgs = [{"role": "user", "content": giant}]
    shards = split_messages(msgs, shard_tokens=3000, max_shards=50, replicate_system=False)
    joined = "".join(m["content"] for s in shards for m in s.messages)
    # 段落边界切分只丢换行，正文片段必须全部保留
    for i in range(2000):
        assert f"第{i}段内容。" in joined


def test_giant_marker_annotates_segment_order():
    giant = "内容。" * 5000
    msgs = [{"role": "user", "content": giant}]
    shards = split_messages(msgs, shard_tokens=3000, max_shards=50, replicate_system=False)
    assert len(shards) > 1
    first = shards[0].messages[0]["content"]
    assert "第 1/" in first and "段" in first


def test_no_duplicate_system_after_giant_split():
    """拆段时 system 不被带进拆段组（prefix 已在最后统一加）。"""
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "内容。" * 5000},
    ]
    shards = split_messages(msgs, shard_tokens=3000, max_shards=50, replicate_system=True)
    for s in shards:
        systems = [m for m in s.messages if m["role"] == "system"]
        assert len(systems) <= 1, "不应出现多个 system"


def test_tool_calls_message_not_split():
    """非纯文本的超大消息（tool_calls）不切内容，单独成片保安全。"""
    big = [{"role": "assistant", "tool_calls": [{"function": {"arguments": "x" * 60000}}]}]
    msgs = [{"role": "user", "content": "小问题"}] + big
    shards = split_messages(msgs, shard_tokens=8000, max_shards=10, replicate_system=False)
    # tool_calls 那条必须原样存在
    flat = [m for s in shards for m in s.messages]
    assert any(m.get("tool_calls") for m in flat)
    assert any("x" * 60000 in str(m.get("tool_calls", "")) for m in flat)
    # 原始那条必须原样保留在某一分片里
    assert any(m.get("tool_calls") and m["tool_calls"] == big[0]["tool_calls"] for m in flat)


# ---------- 文本提取 ----------

def test_extract_text_plain():
    assert _extract_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"


def test_extract_text_parts():
    d = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    assert _extract_text(d) == "a\nb"


def test_extract_text_reasoning_fallback():
    d = {"choices": [{"message": {"reasoning_content": " thinking "}}]}
    assert _extract_text(d) == "thinking"


def test_extract_text_empty():
    assert _extract_text({}) == ""
    assert _extract_text({"choices": []}) == ""


# ---------- KeyPool ----------

def _pool(n=3, **kw):
    return KeyPool(ProviderConfig(name="t", base_url="http://x/v1", keys=[f"k{i}" for i in range(n)]), **kw)


def test_pool_round_robin_returns_distinct_keys():
    p = _pool(3)
    got = [p.acquire()[0] for _ in range(3)]
    assert sorted(got) == [0, 1, 2]


def test_pool_skips_cooling_keys():
    p = _pool(3)
    i = p.acquire()[0]
    p.report(i, False)  # 该 key 进冷却
    a = p.acquire()[0]
    b = p.acquire()[0]
    assert i not in (a, b)


def test_pool_returns_none_when_all_cooling():
    p = _pool(2)
    for i in range(2):
        p.report(i, False)
    assert p.acquire() is None


def test_pool_cooldown_is_exponential_and_capped():
    p = _pool(1, cooldown_seconds=10.0, max_cooldown_seconds=100.0)
    for _ in range(5):
        p.report(0, False)
    left = p._states[0].cooldown_until - __import__("time").monotonic()
    assert left <= 100.0


def test_pool_success_resets_fail_count():
    p = _pool(1)
    p.report(0, False)
    assert p._states[0].fail_count == 1
    p.report(0, True)
    assert p._states[0].fail_count == 0


def test_pool_available_count():
    p = _pool(4)
    assert p.available_count() == 4
    p.report(0, False)
    assert p.available_count() == 3


def test_pool_from_env(monkeypatch):
    monkeypatch.setenv("TEST_KEYS", "aaa,bbb , ccc")
    p = KeyPool.from_provider_dict("t", {"base_url": "http://x", "keys_env": "TEST_KEYS"})
    assert [s.key for s in p._states] == ["aaa", "bbb", "ccc"]


def test_pool_snapshot_masks_keys():
    p = _pool(2)
    snap = p.snapshot()
    for s in snap:
        assert len(s["key_tail"]) <= 6
        assert s["key_tail"] != "k0"


# ---------- 路由决策 ----------

def _router(threshold=1000, shard_tokens=800, max_shards=5):
    pool = _pool(5)
    return ShardRouter(
        pool,
        ShardSettings(token_threshold=threshold, shard_tokens=shard_tokens, max_shards=max_shards),
        AggSettings(reduce_enabled=True),
        {"connect": 5, "read": 10},
    )


def test_decide_passthrough_when_small():
    r = _router(threshold=100000)
    d = r.decide({"messages": [{"role": "user", "content": "短"}]})
    assert d.sharded is False
    assert d.reason == "below_threshold"


def test_decide_shards_when_large():
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    msgs = _mk(turns=40, size=800)
    d = r.decide({"messages": msgs})
    assert d.sharded is True
    assert d.reason == "over_threshold"
    assert d.shard_count > 1


def test_decide_reports_per_shard_tokens():
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    d = r.decide({"messages": _mk(turns=40, size=800)})
    assert len(d.shards) == d.shard_count
    for s in d.shards:
        assert s["tokens"] > 0


# ---------- 端到端（假上游） ----------

class _FakeSession:
    """模拟 aiohttp session：按 key 尾号记录调用，返回固定文本。"""

    def __init__(self, replies=None, fail_tails=()):
        self.calls = []  # [(url, key_tail, body)]
        self.replies = replies or {}
        self.fail_tails = set(fail_tails)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, headers["Authorization"][-4:], json))
        return _FakeCtx(self, url, json, headers)


class _FakeCtx:
    def __init__(self, sess, url, body, headers):
        self.sess, self.url, self.body, self.headers = sess, url, body, headers

    async def __aenter__(self):
        tail = self.headers["Authorization"][-4:]
        if tail in self.sess.fail_tails:
            raise UpstreamError(429, "tpm exceeded")
        return self

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_run_fans_out_to_multiple_keys(monkeypatch):
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    fake = _FakeSession()
    r._session = fake

    async def fake_complete(session, idx, body, ct, rt):
        session.calls.append((session.calls and "x", body))  # noqa
        return {
            "choices": [{"message": {"content": f"part{idx}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    msgs = _mk(turns=40, size=800)
    out = await r.run({"messages": msgs, "temperature": 0.5}, "kimi-k3")
    assert "error" not in out
    xs = out["x_shard_router"]
    assert xs["sharded"] is True
    assert xs["shard_count"] > 1
    assert len(xs["per_shard"]) == xs["shard_count"]


@pytest.mark.asyncio
async def test_run_passthrough_when_small(monkeypatch):
    r = _router(threshold=100000)
    async def fake_complete(session, idx, body, ct, rt):
        return {"choices": [{"message": {"content": "直接回答"}}], "usage": {}}
    monkeypatch.setattr(r.pool, "complete", fake_complete)
    out = await r.run({"messages": [{"role": "user", "content": "小"}]}, "m")
    assert out["x_shard_router"]["sharded"] is False
    assert out["choices"][0]["message"]["content"] == "直接回答"


@pytest.mark.asyncio
async def test_run_uses_reduce_when_enabled(monkeypatch):
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    r.agg.reduce_enabled = True
    calls = {"n": 0}

    async def fake_complete(session, idx, body, ct, rt):
        calls["n"] += 1
        if calls["n"] <= 1:
            return {"choices": [{"message": {"content": "分片结论A"}}], "usage": {}}
        if calls["n"] == 2:
            return {"choices": [{"message": {"content": "分片结论B"}}], "usage": {}}
        return {"choices": [{"message": {"content": "最终汇总"}}], "usage": {}}

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    out = await r.run({"messages": _mk(turns=40, size=800)}, "m")
    assert out["x_shard_router"]["reduce_used"] is True
    assert out["choices"][0]["message"]["content"] == "最终汇总"


@pytest.mark.asyncio
async def test_run_without_reduce_concatenates(monkeypatch):
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    r.agg.reduce_enabled = False
    n = {"i": 0}

    async def fake_complete(session, idx, body, ct, rt):
        n["i"] += 1
        return {"choices": [{"message": {"content": f"片段{n['i']}"}}], "usage": {}}

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    out = await r.run({"messages": _mk(turns=40, size=800)}, "m")
    assert out["x_shard_router"]["reduce_used"] is False
    assert "片段1" in out["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_run_all_shards_fail_returns_error(monkeypatch):
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)

    async def fake_complete(session, idx, body, ct, rt):
        raise UpstreamError(429, "tpm exceeded")

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    out = await r.run({"messages": _mk(turns=40, size=800)}, "m")
    assert "error" in out


@pytest.mark.asyncio
async def test_run_reduce_failure_falls_back_to_concat(monkeypatch):
    r = _router(threshold=1000, shard_tokens=400, max_shards=6)
    n = {"i": 0}

    async def fake_complete(session, idx, body, ct, rt):
        n["i"] += 1
        if n["i"] <= 2:
            return {"choices": [{"message": {"content": f"片{n['i']}"}}], "usage": {}}
        raise UpstreamError(500, "boom")

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    out = await r.run({"messages": _mk(turns=40, size=800)}, "m")
    assert "error" not in out
    assert out["x_shard_router"]["reduce_used"] is False
    assert "片1" in out["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_shards_run_concurrently(monkeypatch):
    """并发性验证：每片 sleep 0.3s，4 片总耗时应远小于 1.2s。"""
    r = _router(threshold=500, shard_tokens=200, max_shards=4)

    async def fake_complete(session, idx, body, ct, rt):
        await asyncio.sleep(0.3)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(r.pool, "complete", fake_complete)
    import time as _t

    t0 = _t.monotonic()
    out = await r.run({"messages": _mk(turns=40, size=600)}, "m")
    dt = _t.monotonic() - t0
    assert "error" not in out
    assert dt < 1.0, f"不够并发，耗时 {dt:.2f}s"


@pytest.mark.asyncio
async def test_openai_shape_fields():
    r = _router(threshold=100000)

    async def fake_complete(session, idx, body, ct, rt):
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3}}

    import app.router as R
    R.ShardRouter._session = None
    orig = r.pool.complete
    r.pool.complete = fake_complete  # type: ignore[method-assign]
    out = await r.run({"messages": [{"role": "user", "content": "x"}]}, "m")
    r.pool.complete = orig  # type: ignore[method-assign]
    for k in ("id", "object", "created", "model", "choices", "usage", "x_shard_router"):
        assert k in out
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["finish_reason"] == "stop"
    # 直连路径 usage 用的是本地 token 估算（上游 usage 只在分片路径汇总）
    assert out["usage"]["prompt_tokens"] == out["usage"]["total_tokens"]
    assert out["usage"]["total_tokens"] > 0
