"""Local single-process lab: isolated run sessions and a legacy manual sandbox."""

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .sessions import RunConfig, SessionManager


def create_app(manager=None):
    sessions = manager or SessionManager(os.environ.get("WEIGHTFORGE_RUNS_DIR", "runs/web"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(sessions.close)

    app = FastAPI(title="WeightForge Lab", lifespan=lifespan)
    app.state.sessions = sessions
    legacy = {}
    legacy_lock = threading.Lock()

    def sandbox():
        from .core import TorchAdapter
        from .demo import dataset, model

        with legacy_lock:
            if not legacy:
                legacy.update(adapter=TorchAdapter(model()), validation=dataset(42, 128))
        return legacy["adapter"], legacy["validation"]

    def get_session(identity):
        try:
            return sessions.get(identity)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "analytical-xor", "scope": "local-demo"}

    @app.get("/inspect")
    def inspect():
        from .demo import observation

        adapter, _ = sandbox()
        return observation(adapter)

    @app.get("/evaluate")
    def metrics():
        from .evaluation import evaluate

        adapter, validation = sandbox()
        return asdict(evaluate(adapter, validation))

    @app.get("/patch")
    def patch():
        adapter, _ = sandbox()
        return adapter.patch()

    @app.post("/proposals")
    def propose(proposal: dict):
        from .evaluation import attempt

        adapter, validation = sandbox()
        try:
            return attempt(adapter, proposal, validation)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/sessions", status_code=201)
    def create_session(config: dict):
        try:
            return sessions.create(RunConfig(**config)).snapshot()
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.get("/sessions/{identity}")
    def session_info(identity: str):
        return get_session(identity).snapshot()

    @app.post("/sessions/{identity}/stop")
    def stop_session(identity: str):
        session = get_session(identity)
        session.cancel.set()
        return session.snapshot()

    @app.get("/sessions/{identity}/results")
    def results(identity: str):
        session = get_session(identity)
        if session.snapshot()["status"] != "completed":
            raise HTTPException(409, detail="Results are available after completion")
        return FileResponse(session.folder / "results.json", media_type="application/json",
                            filename=f"weightforge-{identity}.json")

    @app.get("/sessions/{identity}/events")
    async def events(identity: str, request: Request, after: int = 0):
        session = get_session(identity)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(422, detail="Invalid event cursor") from exc
        if cursor < 0 or cursor > len(session.events):
            raise HTTPException(422, detail="Invalid event cursor")

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows, done = await asyncio.to_thread(session.wait_events, cursor)
                for row in rows:
                    yield f"id: {row['id']}\nevent: update\ndata: {json.dumps(row, allow_nan=False)}\n\n"
                    cursor = row["id"]
                if done:
                    return
                if not rows:
                    yield ": heartbeat\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(static / "index.html")

    return app
