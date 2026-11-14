"""Local single-process lab: isolated run sessions and a legacy manual sandbox."""

import asyncio
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .sessions import RunConfig, SessionManager


def create_app(manager=None):
    sessions = manager or SessionManager(os.environ.get("WEIGHTFORGE_RUNS_DIR", "runs/web"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(sessions.close)

    app = FastAPI(title="WeightForge Lab", lifespan=lifespan, root_path="/api")
    app.state.sessions = sessions
    model_root = Path(os.environ.get("WEIGHTFORGE_MODELS_DIR", "runs/models"))

    @app.get("/capabilities")
    def capabilities():
        from .mnist import data_root

        ready = all((data_root() / name).exists() for name in (
            "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"))
        return {"providers": ["demo", "guided", "random", "openai"],
                "llm_configured": bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("WEIGHTFORGE_LLM_MODEL")),
                "llm_model": os.environ.get("WEIGHTFORGE_LLM_MODEL", ""),
                "mnist_data_ready": ready, "supported_architectures": ["mnist-mlp-64"],
                "max_upload_bytes": 2000000, "deployment": "local-single-user"}

    def artifact(identity, name):
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise HTTPException(404, "Unknown experiment")
        path = sessions.root / identity / name
        if not path.is_file():
            raise HTTPException(404, "Artifact not available")
        return path

    @app.get("/experiments/{identity}")
    def saved_result(identity: str):
        return json.loads(artifact(identity, "results.json").read_text(encoding="utf-8"))

    @app.get("/experiments/{identity}/checkpoint")
    def checkpoint(identity: str):
        artifact(identity, "results.json")
        return FileResponse(artifact(identity, "model.pt"), filename=f"weightforge-{identity}.pt")

    @app.get("/experiments/{identity}/download")
    def saved_download(identity: str):
        return FileResponse(artifact(identity, "results.json"), filename=f"weightforge-{identity}.json")

    @app.get("/models")
    def models():
        rows = []
        for path in sorted(model_root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                name = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["name"]
            except (OSError, ValueError, KeyError):
                name = "MNIST MLP"
            rows.append({"id": path.stem, "name": name, "bytes": path.stat().st_size})
        return rows

    @app.post("/models", status_code=201)
    async def import_model(request: Request):
        from .mnist import load_checkpoint

        model_root.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        path = model_root / (identity + ".upload")
        size = 0
        try:
            with path.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 2_000_000:
                        raise ValueError("Maximum checkpoint size is 2 MB")
                    output.write(chunk)
            await asyncio.to_thread(load_checkpoint, path)
            name = unquote(request.headers.get("X-Model-Name", "MNIST MLP"))[:80]
            name = "".join(c for c in name if c.isprintable()).strip() or "MNIST MLP"
            (model_root / (identity + ".json")).write_text(json.dumps({"name": name}), encoding="utf-8")
            path.rename(model_root / (identity + ".pt"))
            return {"id": identity, "name": name, "bytes": size}
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, detail="Checkpoint rejected: expected a valid MNIST MLP float32 state_dict") from exc

    @app.get("/experiments")
    def saved_experiments():
        rows = []
        for path in sorted(sessions.root.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append({"id": path.parent.name, "config": result["config"],
                         "initial": result["initial"], "retained": result["retained"]})
        return rows
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
        return {"status": "ok", "scope": "local-single-user"}

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
            parsed = RunConfig(**config)
            if parsed.provider == "openai" and not capabilities()["llm_configured"]:
                raise ValueError("Configure the LLM provider in backend environment first")
            if parsed.model_id != "xor":
                if not (model_root / f"{parsed.model_id}.pt").is_file():
                    raise ValueError("Import a supported checkpoint first")
                if not capabilities()["mnist_data_ready"]:
                    raise ValueError("MNIST data missing. Run python -m weightforge.mnist --download")
            return sessions.create(parsed).snapshot()
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
            previous = int(request.headers.get("last-event-id", "0"))
            if after < 0 or previous < 0:
                raise ValueError("Negative cursor")
            cursor = max(after, previous)
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

    return app
