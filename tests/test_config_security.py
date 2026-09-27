"""安全与配置合并的回归测试：重点锁死「key 不被泄露」和「key 不被清空」。

这两个是真实出现过的坑：
- /api/status 曾把 providers_raw 整体吐出，含明文 key；
- UI「保存配置」曾用 status 拿到的数据整体回写，把 key 静默清空。
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def _write_cfg(path: Path, keys: list[str]) -> str:
    doc = {
        "server": {"host": "127.0.0.1", "port": 20200},
        "providers": {
            "sensenova": {
                "name": "SenseNova",
                "base_url": "https://token.sensenova.cn/v1",
                "keys": keys,
                "models": ["kimi-k3"],
            }
        },
        "sharding": {"token_threshold": 100, "shard_tokens": 50, "max_shards": 3,
                     "replicate_system": True, "overlap_ratio": 0.1},
        "aggregation": {"reduce_enabled": True, "reduce_mode": "flat"},
        "timeouts": {"connect": 5, "read": 10},
    }
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture()
def cfg_path(tmp_path):
    return tmp_path / "config.yaml"


def _load_server(cfg: str):
    os.environ["SHARD_ROUTER_CONFIG"] = cfg
    if "server" in sys.modules:
        del sys.modules["server"]
    import server as srv
    importlib.reload(srv)
    return srv


# ---------- key 不泄露 ----------

def test_status_never_returns_plaintext_keys(cfg_path):
    keys = ["sk-SECRETVALUE0000000001", "sk-SECRETVALUE0000000002"]
    srv = _load_server(_write_cfg(cfg_path, keys))
    data = asyncio.run(srv.status())
    blob = json.dumps(data)
    for k in keys:
        assert k not in blob, f"明文 key 泄露到 /api/status: {k[:12]}…"
    # 但计数要保留，前端要靠它显示
    raw = data["providers_raw"]["sensenova"]
    assert raw["keys"] == []
    assert raw["_key_count"] == 2
    assert data["providers"]["sensenova"]["key_count"] == 2


def test_status_masks_keys_from_env_too(cfg_path, monkeypatch):
    keys = ["sk-ENVSECRET0000000001", "sk-ENVSECRET0000000002", "sk-ENVSECRET00000000003"]
    monkeypatch.setenv("MYSECRETKEYS", ",".join(keys))
    p = cfg_path
    doc = {
        "server": {"host": "127.0.0.1", "port": 20200},
        "providers": {
            "p": {"base_url": "https://x/v1", "keys": [], "keys_env": "MYSECRETKEYS",
                  "models": ["m"]}
        },
        "sharding": {}, "aggregation": {}, "timeouts": {},
    }
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    srv = _load_server(str(p))
    data = asyncio.run(srv.status())
    blob = json.dumps(data)
    for k in keys:
        assert k not in blob, "keys_env 提供的明文 key 也不得泄露"


def test_pool_snapshot_only_shows_tail():
    """pool.snapshot() 只能给尾号。"""
    sys.path.insert(0, str(BASE))
    from app.pool import KeyPool, ProviderConfig

    p = KeyPool(ProviderConfig("t", "http://x/v1", ["sk-ABCDEF0123456789", "sk-XYZ987"]))
    blob = json.dumps(p.snapshot())
    assert "sk-ABCDEF0123456789" not in blob
    assert "sk-XYZ987" not in blob
    for s in p.snapshot():
        assert len(s["key_tail"]) <= 6


# ---------- key 不被清空 ----------

class _FakeReq:
    def __init__(self, body):
        self._b = body

    async def json(self):
        return self._b


def test_config_update_preserves_keys_when_omitted(cfg_path):
    """核心回归：UI 保存策略时不带 keys，磁盘上的 key 必须原样保留。"""
    keys = ["sk-KEEPME000000000001", "sk-KEEPME000000000002"]
    srv = _load_server(_write_cfg(cfg_path, keys))
    assert len(srv.SETTINGS.pools["sensenova"]._states) == 2

    body = {
        "server": srv.SETTINGS.server,
        # 模拟经 /api/status 抹空后回写的 providers
        "providers": {"sensenova": {"name": "SenseNova",
                                    "base_url": "https://token.sensenova.cn/v1",
                                    "keys": [], "models": ["kimi-k3"]}},
        "sharding": {"token_threshold": 999, "shard_tokens": 123, "max_shards": 3,
                     "replicate_system": True, "overlap_ratio": 0.2},
        "aggregation": {"reduce_enabled": True, "reduce_mode": "tree"},
        "timeouts": srv.SETTINGS.timeouts,
    }
    res = asyncio.run(srv.update_config(_FakeReq(body)))
    d = json.loads(res.body)
    assert d.get("ok") is True, d
    # key 还在
    disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert disk["providers"]["sensenova"]["keys"] == keys, "key 被静默清空了！"
    # 策略确实改了
    assert disk["sharding"]["token_threshold"] == 999


def test_config_update_overrides_keys_when_explicit(cfg_path):
    """显式带 keys 时必须覆盖（这是 UI 填 key 的路径）。"""
    old = ["sk-OLD00000000000001"]
    srv = _load_server(_write_cfg(cfg_path, old))
    new_keys = ["sk-NEW00000000000001", "sk-NEW00000000000002"]
    body = {
        "server": srv.SETTINGS.server,
        "providers": {"sensenova": {"keys": new_keys, "keys_env": None}},
        "sharding": srv.SETTINGS.sharding,
        "aggregation": srv.SETTINGS.aggregation,
        "timeouts": srv.SETTINGS.timeouts,
    }
    res = asyncio.run(srv.update_config(_FakeReq(body)))
    assert json.loads(res.body).get("ok") is True
    disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert disk["providers"]["sensenova"]["keys"] == new_keys
    # keys 生效
    assert len(srv.SETTINGS.pools["sensenova"]._states) == 2


def test_config_update_creates_backup(cfg_path):
    srv = _load_server(_write_cfg(cfg_path, ["sk-B1"]))
    body = {"sharding": srv.SETTINGS.sharding, "aggregation": srv.SETTINGS.aggregation,
            "timeouts": srv.SETTINGS.timeouts, "server": srv.SETTINGS.server}
    asyncio.run(srv.update_config(_FakeReq(body)))
    baks = list(cfg_path.parent.glob("config.yaml.bak-*"))
    assert baks, "保存前应备份 config.yaml"


def test_config_update_bad_body_does_not_destroy_file(cfg_path):
    keys = ["sk-SURVIVE00000000001"]
    _write_cfg(cfg_path, keys)
    srv = _load_server(str(cfg_path))
    before = cfg_path.read_text(encoding="utf-8")
    res = asyncio.run(srv.update_config(_FakeReq({"providers": "这不是字典"})))
    after = cfg_path.read_text(encoding="utf-8")
    # 磁盘不应被写坏
    assert "sk-SURVIVE00000000001" in after, "异常输入把 key 冲掉了"
    assert before == after or after, "配置文件不应被破坏"


def test_deep_merge_leaves_unknown_provider_keys_untouched(cfg_path):
    """provider 的其他字段（base_url/models）不在请求体里也得保留。"""
    keys = ["sk-P1"]
    _write_cfg(cfg_path, keys)
    srv = _load_server(str(cfg_path))
    body = {"providers": {"sensenova": {"models": ["kimi-k3", "glm-5.2"]}},
            "sharding": srv.SETTINGS.sharding,
            "aggregation": srv.SETTINGS.aggregation,
            "timeouts": srv.SETTINGS.timeouts,
            "server": srv.SETTINGS.server}
    res = asyncio.run(srv.update_config(_FakeReq(body)))
    assert json.loads(res.body).get("ok") is True
    disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = disk["providers"]["sensenova"]
    assert p["keys"] == keys, "key 被清了"
    assert p["base_url"] == "https://token.sensenova.cn/v1", "base_url 被清了"
    assert p["models"] == ["kimi-k3", "glm-5.2"], "models 没更新"
