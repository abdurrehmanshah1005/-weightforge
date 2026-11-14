"""Local single-process demo service. One adapter; all access serialized."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from .core import TorchAdapter
from .demo import dataset, model, observation
from .evaluation import attempt, evaluate


def create_app():
    app = FastAPI(title="WeightForge MVP")
    adapter = TorchAdapter(model())
    validation = dataset(42, 128)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "analytical-xor", "scope": "local-demo"}

    @app.get("/inspect")
    def inspect():
        return observation(adapter)

    @app.get("/evaluate")
    def metrics():
        return asdict(evaluate(adapter, validation))

    @app.get("/patch")
    def patch():
        return adapter.patch()

    @app.post("/proposals")
    def propose(proposal: dict):
        try:
            return attempt(adapter, proposal, validation)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
