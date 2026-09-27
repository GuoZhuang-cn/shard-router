"""Provider 池：把多个 key 组成一个池，支持轮询 + 失败冷却。

设计原则（来自对 SenseNova TPM 限流的实测）：
- TPM 是**分钟级 token 配额**，多 key 并发能绕过单 key 的分钟配额。
- 但瞬时并发太高仍然会撞 RPM（每分钟请求数），所以用 shard 数量
  （= 并发数）本身作为天然的节流：一次请求最多 max_shards 个并发。
- 单 key 撞限（429/5xx/超时）后冷却 cooldown 秒，期间不再分配。
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp


@dataclass
class KeyState:
    key: str
    cooldown_until: float = 0.0
    fail_count: int = 0
    ok_count: int = 0
    total_tokens: int = 0

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    keys: list[str]
    models: list[str] = field(default_factory=list)


class KeyPool:
    """一个 provider 的 key 池。线程安全的轮询 + 冷却。"""

    def __init__(
        self,
        cfg: ProviderConfig,
        cooldown_seconds: float = 30.0,
        max_cooldown_seconds: float = 300.0,
    ) -> None:
        self.cfg = cfg
        self.cooldown_seconds = cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds
        self._states = [KeyState(key=k) for k in cfg.keys]
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(self._states))) if self._states else None

    @classmethod
    def from_provider_dict(cls, name: str, d: dict[str, Any], **kw: Any) -> "KeyPool":
        keys = list(d.get("keys") or [])
        env = d.get("keys_env")
        if env and not keys:
            raw = os.environ.get(env, "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        cfg = ProviderConfig(
            name=d.get("name", name),
            base_url=d["base_url"].rstrip("/"),
            keys=keys,
            models=list(d.get("models") or []),
        )
        return cls(cfg, **kw)

    def available_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for s in self._states if s.available(now))

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "key_tail": s.key[-6:] if len(s.key) > 6 else "***",
                    "cooling": not s.available(now),
                    "cooldown_left": round(max(0.0, s.cooldown_until - now), 1),
                    "ok": s.ok_count,
                    "fail": s.fail_count,
                    "tokens": s.total_tokens,
                }
                for s in self._states
            ]

    def acquire(self) -> tuple[int, KeyState] | None:
        """拿一个可用 key 的下标和状态；全冷却返回 None。"""
        if not self._states:
            return None
        now = time.monotonic()
        with self._lock:
            # 轮询一圈内找第一个可用的
            if self._cycle is None:
                return None
            n = len(self._states)
            for _ in range(n):
                i = next(self._cycle)
                s = self._states[i]
                if s.available(now):
                    return i, s
            return None

    def report(self, index: int, ok: bool, tokens: int = 0, retry_after: float | None = None) -> None:
        with self._lock:
            s = self._states[index]
            s.total_tokens += tokens
            if ok:
                s.ok_count += 1
                s.fail_count = 0
            else:
                s.fail_count += 1
                if retry_after:
                    # 上游明确给了 Retry-After 就照办，不用 max_cooldown 砍它
                    cd = max(0.5, float(retry_after))
                else:
                    # 指数退避：30 → 60 → 120 → … 封顶 max_cooldown
                    cd = min(
                        self.cooldown_seconds * (2 ** min(s.fail_count - 1, 5)),
                        self.max_cooldown_seconds,
                    )
                s.cooldown_until = time.monotonic() + cd

    def seconds_until_next_available(self) -> float | None:
        """最快恢复的 key 还要几秒。全可用返回 0，无 key 返回 None。"""
        if not self._states:
            return None
        now = time.monotonic()
        with self._lock:
            lefts = [max(0.0, s.cooldown_until - now) for s in self._states]
        return min(lefts)

    async def complete(
        self,
        session: aiohttp.ClientSession,
        index: int,
        payload: dict[str, Any],
        connect_timeout: float,
        read_timeout: float,
    ) -> dict[str, Any]:
        """向单个 key 发一次 chat/completions，返回解析后的 JSON。"""
        s = self._states[index]
        timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_read=read_timeout)
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {s.key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",  # 防 gzip 攒缓冲拖慢 TTFT
        }
        body = dict(payload)
        body.setdefault("stream", False)
        # 仅在客户端没指定时才注入 usage 统计，尊重显式的 false
        if body.get("stream") and not isinstance(body.get("stream_options"), dict):
            body["stream_options"] = {"include_usage": True}

        async with session.post(
            url, json=body, headers=headers, timeout=timeout
        ) as resp:
            # 上游给了 Retry-After 就照办（SenseNova 目前不给，但要兼容给了的实现）
            ra = None
            try:
                ra_raw = resp.headers.get("Retry-After")
                if ra_raw:
                    ra = float(str(ra_raw).strip())
            except (TypeError, ValueError):
                ra = None
            raw = await resp.read()
            if resp.status >= 400:
                raise UpstreamError(resp.status, raw[:400].decode("utf-8", "replace"), retry_after=ra)
            try:
                data = raw and __import__("json").loads(raw) or {}
            except Exception as e:  # noqa: BLE001
                raise UpstreamError(resp.status, f"bad json: {e}; head={raw[:200]!r}")
            usage = data.get("usage") or {}
            self.report(index, True, int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)))
            return data


class UpstreamError(Exception):
    def __init__(self, status: int, detail: str, retry_after: float | None = None) -> None:
        super().__init__(f"upstream {status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after
