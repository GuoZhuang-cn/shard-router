"""shard-router 服务入口。

端点：
  POST /v1/chat/completions   OpenAI 兼容（9router 把它配成一个 provider 的 base_url）
  GET  /api/status             UI 用：池子/配置/统计
  POST /api/config             UI 用：改配置并热加载
  POST /api/test               UI 用：发一次探测请求
  GET  /                      Web UI
  GET  /healthz               健康检查
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.config import Settings, load, reload as reload_settings  # noqa: E402
from app.pool import KeyPool, UpstreamError  # noqa: E402
from app.router import AggSettings, RetrySettings, ShardRouter, ShardSettings  # noqa: E402

CONFIG_PATH = os.environ.get("SHARD_ROUTER_CONFIG", str(BASE / "config.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("shard-router")

app = FastAPI(title="shard-router", version="0.1.0")

SETTINGS: Settings = load(CONFIG_PATH)
ROUTERS: dict[str, ShardRouter] = {}
STATS: dict[str, Any] = {
    "requests": 0,
    "sharded": 0,
    "ok": 0,
    "failed": 0,
    "started_at": time.time(),
}


def _build_routers(st: Settings) -> dict[str, ShardRouter]:
    out: dict[str, ShardRouter] = {}
    sh = st.sharding or {}
    ag = st.aggregation or {}
    rt = st.retry or {}
    for name, pool in st.pools.items():
        out[name] = ShardRouter(
            pool=pool,
            settings=ShardSettings(
                token_threshold=int(sh.get("token_threshold", 12000)),
                shard_tokens=int(sh.get("shard_tokens", 8000)),
                max_shards=int(sh.get("max_shards", 10)),
                replicate_system=bool(sh.get("replicate_system", True)),
                overlap_ratio=float(sh.get("overlap_ratio", 0.0)),
            ),
            agg=AggSettings(
                mode=ag.get("mode", "stream"),
                reduce_enabled=bool(ag.get("reduce_enabled", True)),
                reduce_model=ag.get("reduce_model"),
                reduce_mode=ag.get("reduce_mode", "flat"),
            ),
            timeouts=st.timeouts,
            retry=RetrySettings(
                max_attempts=int(rt.get("max_attempts", 8)),
                rate_limit_backoff=float(rt.get("rate_limit_backoff", 2.0)),
                max_backoff=float(rt.get("max_backoff", 30.0)),
                max_wait_for_key=float(rt.get("max_wait_for_key", 20.0)),
            ),
        )
    return out


ROUTERS = _build_routers(SETTINGS)


@app.on_event("startup")
async def _startup() -> None:
    for r in ROUTERS.values():
        await r.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    for r in ROUTERS.values():
        await r.stop()


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "uptime": round(time.time() - STATS["started_at"], 1)}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    providers = {}
    for name, pool in SETTINGS.pools.items():
        providers[name] = {
            "name": pool.cfg.name,
            "base_url": pool.cfg.base_url,
            "models": pool.cfg.models,
            "key_count": len(pool._states),
            "available": pool.available_count(),
            "keys": pool.snapshot(),
        }
    def _mask_providers(raw: dict[str, Any]) -> dict[str, Any]:
        """深拷贝 providers 配置并把明文 keys 换成计数占位。

        /api/status 是 UI 拉的接口，而这个服务裸奔在公网 IP 上（无鉴权）。
        前端只需要知道「有几个 key、从哪来」，绝不能让完整 key 出现在
        任何 HTTP 响应里。回写配置时用的是不含 keys 的骨架 + 用户新填的值。
        """
        out: dict[str, Any] = {}
        for name, p in (raw or {}).items():
            q = dict(p)
            if "keys" in q:
                q["keys"] = []
                q["_key_count"] = len(p.get("keys") or [])
            if "keys_env" in q and q.get("keys_env"):
                q["_key_count"] = len(SETTINGS.pools[name]._states) if name in SETTINGS.pools else 0
            out[name] = q
        return out

    return {
        "providers": providers,
        "providers_raw": _mask_providers(SETTINGS.raw.get("providers") or {}),
        "server": SETTINGS.server,
        "sharding": SETTINGS.sharding,
        "aggregation": SETTINGS.aggregation,
        "timeouts": SETTINGS.timeouts,
        "retry": SETTINGS.retry,
        "stats": STATS,
        "config_path": CONFIG_PATH,
    }


@app.post("/api/config")
async def update_config(request: Request) -> JSONResponse:
    """UI 保存配置。

    **部分更新语义**：请求体里某个 provider 若不含 `keys` 字段，则保留磁盘上
    已有的 key。这是必要的——前端拿到的 providers_raw 里 keys 已被抹成 []
    （不能把明文 key 吐给无鉴权的 /api/status），如果整体回写就会静默清空
    用户填好的 key。只有显式带了 keys（可能为空数组）时才覆盖。

    保存前自动备份 config.yaml 到 config.yaml.bak-<ts>。
    """
    global SETTINGS, ROUTERS
    body = await request.json()
    try:
        # 合并：磁盘现值 ← 请求体（per-provider 深合并）
        cur = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8")) or {}
        cur = _deep_merge_providers(cur, body)

        import shutil, time as _t
        bak = f"{CONFIG_PATH}.bak-{int(_t.time())}"
        shutil.copy2(CONFIG_PATH, bak)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cur, f, allow_unicode=True, sort_keys=False)

        new = reload_settings(SETTINGS, CONFIG_PATH)
        SETTINGS = new
        ROUTERS = _build_routers(new)
        for r in ROUTERS.values():
            await r.start()
        return JSONResponse({"ok": True, "message": f"配置已保存并热加载（备份 {bak}）"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": f"{type(e).__name__}: {e}"}, status_code=400)


def _deep_merge_providers(cur: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """把请求体合并进磁盘配置。

    providers 段逐字段合并：`keys` 字段只在请求体显式提供时才覆盖，
    否则保留磁盘现值。其余段（server/sharding/aggregation/timeouts）整体替换。
    """
    out = dict(cur)
    for k, v in (body or {}).items():
        if k == "providers":
            merged: dict[str, Any] = {}
            cur_p = out.get("providers") or {}
            for name, p in (v or {}).items():
                base = dict(cur_p.get(name) or {})
                for pk, pv in (p or {}).items():
                    if pk == "keys":
                        # 空数组视为「未提供」而不是「清空」。
                        # 前端看到的是被抹空的 providers_raw，若把空数组当成显式清空，
                        # 用户填好的 key 会在下一次「保存配置」时静默丢失。
                        if pv:
                            base["keys"] = pv
                    elif pk == "keys_env" and pv:
                        base["keys_env"] = pv
                    else:
                        base[pk] = pv
                # 填了直存 keys 就让 keys_env 失效：否则 env 优先，
                # 用户在 UI 填的 key 会看起来没生效
                if base.get("keys"):
                    base["keys_env"] = None
                merged[name] = base
            out["providers"] = merged
        else:
            out[k] = v
    return out


@app.post("/api/test")
async def test_provider(request: Request) -> JSONResponse:
    """UI 的「测试连通」按钮：向指定 provider 发一条小请求。"""
    body = await request.json()
    pname = body.get("provider")
    model = body.get("model")
    router = ROUTERS.get(pname)
    if router is None:
        return JSONResponse({"ok": False, "message": f"unknown provider {pname}"}, status_code=404)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": body.get("prompt", "说三个字：你好呀")}],
    }
    try:
        await router.start()
        result = await router.run(payload, model)
        dec = router.decide(payload)
        return JSONResponse({
            "ok": "error" not in result,
            "sharded": dec.sharded,
            "est_tokens": dec.est_tokens,
            "reply": (result.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            "error": result.get("error"),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI 兼容的模型列表。

    9router 把这个服务当 provider 探测时会打这个端点；没有它就报 404，
    provider 加不进来。按所有 provider 的 models 去重汇总。
    """
    seen: dict[str, str] = {}
    for r in ROUTERS.values():
        owner = r.pool.cfg.name
        for m in (r.pool.cfg.models or []):
            seen.setdefault(m, owner)
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": owner}
            for m, owner in sorted(seen.items())
        ],
    }


def _pick_router(model: str) -> ShardRouter | None:
    """按 model 名找 provider；找不到就用第一个。"""
    for r in ROUTERS.values():
        if model in (r.pool.cfg.models or []):
            return r
    return next(iter(ROUTERS.values()), None)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    STATS["requests"] += 1
    body = await request.json()
    model = body.get("model") or ""
    router = _pick_router(model)
    if router is None:
        return JSONResponse({"error": {"message": "no provider configured", "code": 500}}, status_code=500)

    dec = router.decide(body)
    if dec.sharded:
        STATS["sharded"] += 1
    log.info(
        "REQ model=%s est_tokens=%d sharded=%s shards=%d reason=%s",
        model, dec.est_tokens, dec.sharded, dec.shard_count, dec.reason,
    )

    if body.get("stream"):
        # 分片内部是并发的，无法真正逐 token 流给客户端；
        # 用「一次性」OpenAI chunk 流模拟，保证客户端兼容。
        try:
            await router.start()
            result = await router.run(body, model)
        except Exception as e:  # noqa: BLE001
            STATS["failed"] += 1
            return JSONResponse(
                {"error": {"message": f"{type(e).__name__}: {e}", "code": 502}}, status_code=502
            )
        if "error" in result:
            STATS["failed"] += 1
            return JSONResponse(result, status_code=502)
        STATS["ok"] += 1

        def gen():
            text = result["choices"][0]["message"]["content"]
            chunk = {
                "id": result["id"],
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {
                "id": result["id"],
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        await router.start()
        result = await router.run(body, model)
    except UpstreamError as e:
        STATS["failed"] += 1
        return JSONResponse({"error": {"message": str(e), "code": e.status}}, status_code=502)
    except Exception as e:  # noqa: BLE001
        STATS["failed"] += 1
        return JSONResponse({"error": {"message": f"{type(e).__name__}: {e}", "code": 502}}, status_code=502)

    if "error" in result:
        STATS["failed"] += 1
        return JSONResponse(result, status_code=502)
    STATS["ok"] += 1
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    p = BASE / "static" / "index.html"
    return p.read_text(encoding="utf-8") if p.exists() else "<h1>shard-router</h1><p>static/index.html missing</p>"


def main() -> None:
    srv = SETTINGS.server or {}
    uvicorn.run(
        app,
        host=srv.get("host", "0.0.0.0"),
        port=int(srv.get("port", 20200)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
