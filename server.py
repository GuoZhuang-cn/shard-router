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
from app.router import AggSettings, ShardRouter, ShardSettings  # noqa: E402

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
    for name, pool in st.pools.items():
        out[name] = ShardRouter(
            pool=pool,
            settings=ShardSettings(
                token_threshold=int(sh.get("token_threshold", 12000)),
                shard_tokens=int(sh.get("shard_tokens", 8000)),
                max_shards=int(sh.get("max_shards", 10)),
                replicate_system=bool(sh.get("replicate_system", True)),
            ),
            agg=AggSettings(
                mode=ag.get("mode", "stream"),
                reduce_enabled=bool(ag.get("reduce_enabled", True)),
                reduce_model=ag.get("reduce_model"),
            ),
            timeouts=st.timeouts,
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
    return {
        "providers": providers,
        "providers_raw": SETTINGS.raw.get("providers") or {},
        "server": SETTINGS.server,
        "sharding": SETTINGS.sharding,
        "aggregation": SETTINGS.aggregation,
        "timeouts": SETTINGS.timeouts,
        "stats": STATS,
        "config_path": CONFIG_PATH,
    }


@app.post("/api/config")
async def update_config(request: Request) -> JSONResponse:
    """UI 保存配置：全量替换 config.yaml 后热加载。"""
    global SETTINGS, ROUTERS
    body = await request.json()
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(body, f, allow_unicode=True, sort_keys=False)
        new = reload_settings(SETTINGS, CONFIG_PATH)
        SETTINGS = new
        ROUTERS = _build_routers(new)
        for r in ROUTERS.values():
            await r.start()
        return JSONResponse({"ok": True, "message": "配置已保存并热加载"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": f"{type(e).__name__}: {e}"}, status_code=400)


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
