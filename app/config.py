"""配置加载。

provider 支持多套；UI 只改配置后热加载，不重启进程。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from .pool import KeyPool


@dataclass
class Settings:
    raw: dict[str, Any]
    pools: dict[str, KeyPool] = field(default_factory=dict)

    @property
    def server(self) -> dict[str, Any]:
        return self.raw.get("server") or {}

    @property
    def sharding(self) -> dict[str, Any]:
        return self.raw.get("sharding") or {}

    @property
    def aggregation(self) -> dict[str, Any]:
        return self.raw.get("aggregation") or {}

    @property
    def timeouts(self) -> dict[str, Any]:
        return self.raw.get("timeouts") or {}

    @property
    def logging(self) -> dict[str, Any]:
        return self.raw.get("logging") or {}


def load(path: str) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    st = Settings(raw=raw)
    providers = raw.get("providers") or {}
    for name, p in providers.items():
        kw = {}
        st.pools[name] = KeyPool.from_provider_dict(name, p, **kw)
    return st


def reload(st: Settings, path: str) -> Settings:
    """重新读盘，保留已有 pool 的计数（key 集合没变时）。"""
    new = load(path)
    for name, pool in new.pools.items():
        old = st.pools.get(name)
        if old and [s.key for s in old._states] == [s.key for s in pool._states]:
            pool._states = old._states
    return new
