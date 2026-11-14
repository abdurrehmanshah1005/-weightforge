"""Bounded single-worker sessions, replayable events, and cooperative cancellation.

The registry is process-local. Completed JSON artifacts survive restarts; live
model state does not. The runner is injected to test lifecycle without a GPU.
"""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    model_id: str = "xor"
    provider: str = "demo"
    llm_consent: bool = False
    budget: int = 12
    seed: int = 42
    random_runs: int = 3
    max_delta: float = 2.0
    max_changed: int = 8

    def __post_init__(self):
        if self.model_id != "xor":
            if (type(self.model_id) is not str or len(self.model_id) != 32
                    or any(c not in "0123456789abcdef" for c in self.model_id)):
                raise ValueError("Invalid model ID")
            if self.provider == "demo":
                raise ValueError("The scripted demonstration is XOR-only")
        if self.provider not in ("demo", "guided", "random", "openai"):
            raise ValueError("Unknown proposal provider")
        if type(self.llm_consent) is not bool or (self.provider == "openai" and not self.llm_consent):
            raise ValueError("LLM runs require explicit consent to sending observations and API charges")
        for name, low, high in (("budget", 1, 50), ("seed", 0, 2**31 - 1),
                                ("random_runs", 1, 10), ("max_changed", 1, 9)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        if type(self.max_delta) not in (int, float) or not 0 < self.max_delta <= 2:
            raise ValueError("max_delta must be in (0, 2]")


@dataclass
class Session:
    id: str
    config: RunConfig
    folder: Path
    status: str = "queued"
    events: list = field(default_factory=list)
    result: dict | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    started: float = field(default_factory=time.monotonic)

    def emit(self, stage, **payload):
        with self.condition:
            event = {"id": len(self.events) + 1, "stage": stage,
                     "elapsed_ms": round((time.monotonic() - self.started) * 1000), **payload}
            self.events.append(event)
            self.condition.notify_all()
            return event

    def snapshot(self):
        with self.condition:
            return {"id": self.id, "status": self.status, "config": asdict(self.config),
                    "event_count": len(self.events), "result": self.result}

    def wait_events(self, after, timeout=10):
        with self.condition:
            if len(self.events) <= after and self.status not in TERMINAL:
                self.condition.wait(timeout)
            return list(self.events[after:]), self.status in TERMINAL


TERMINAL = {"completed", "cancelled", "failed"}


class Cancelled(Exception):
    pass


class SessionManager:
    def __init__(self, root, runner=None, capacity=20):
        self.root = Path(root)
        self.runner = runner or run_experiment
        self.capacity = capacity
        self.sessions = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weightforge")

    def create(self, config):
        with self.lock:
            # Bound queued work as well as retained process memory.
            if sum(s.status not in TERMINAL for s in self.sessions.values()) >= 2:
                raise ValueError("Two experiments are already active or queued")
            while len(self.sessions) >= self.capacity:
                victim = next((key for key, s in self.sessions.items() if s.status in TERMINAL), None)
                if victim is None:
                    raise ValueError("Session capacity reached")
                del self.sessions[victim]
            identity = uuid.uuid4().hex
            folder = self.root / identity
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "config.json").write_text(
                json.dumps(asdict(config), indent=2, allow_nan=False), encoding="utf-8")
            session = Session(identity, config, folder)
            self.sessions[identity] = session
            session.emit("queued", message="Waiting for the experiment worker")
            self.pool.submit(self._execute, session)
            return session

    def get(self, identity):
        with self.lock:
            if identity not in self.sessions:
                raise KeyError("Experiment not found; it may have expired or the server restarted")
            return self.sessions[identity]

    def _execute(self, session):
        try:
            if session.cancel.is_set():
                raise Cancelled()
            with session.condition:
                session.status = "running"
            result = self.runner(session)
            # Save the exact result before announcing its terminal state.
            (session.folder / "results.tmp").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            (session.folder / "results.tmp").replace(session.folder / "results.json")
            with session.condition:
                session.result = result
                session.status = "completed"
                session.emit("completed", result=result)
        except Cancelled:
            with session.condition:
                session.status = "cancelled"
                session.emit("cancelled", message="Stopped between operations; no edit left pending")
        except Exception as exc:
            with session.condition:
                session.status = "failed"
                session.emit("failed", message=str(exc))
        finally:
            (session.folder / "events.json").write_text(
                json.dumps(session.events, indent=2, allow_nan=False), encoding="utf-8")

    def close(self):
        with self.lock:
            for session in self.sessions.values():
                session.cancel.set()
        self.pool.shutdown(wait=True)


def run_experiment(session, *, checkpoint=None, holdout_start=0):
    """Real PyTorch work. No sleeps, invented metrics or full model copies."""
    import hashlib
    import platform

    import torch

    from .core import Limits, TorchAdapter
    from .evaluation import evaluate
    from .providers import CalibrationProvider, OpenAIProvider, RandomProvider
    from .tasks import Task

    config = session.config
    task = Task(config.model_id, config.seed, checkpoint, holdout_start)
    model = task.factory

    def observation(current):
        return {"task": task.name, "inspection": current.inspect(),
                "max_delta": current.limits.max_delta,
                "diagnostics": task.diagnostics(current)}
    limits = Limits(max_edits=1, max_delta=config.max_delta, max_changed=config.max_changed)
    adapter = TorchAdapter(model(), limits)
    validation = task.validation
    holdout = task.holdout

    def check_cancel():
        if session.cancel.is_set():
            raise Cancelled()

    def metrics(current, data=validation):
        return asdict(evaluate(current, data))

    initial = metrics(adapter)
    observation_size = len(json.dumps(observation(adapter)).encode())
    digest = hashlib.sha256()
    for name, p in adapter.parameters.items():
        digest.update(name.encode())
        digest.update(json.dumps(p.detach().flatten().tolist()).encode())
    session.emit("started", initial=initial, model={"name": task.name, "parameters": sum(p.numel() for p in adapter.parameters.values()),
                 "weight_bytes": sum(p.numel() * p.element_size() for p in adapter.parameters.values()),
                 "observation_bytes": observation_size, "context_limit": limits.context_bytes},
                 message="Fresh model loaded; validation measured")

    def trial(current, provider, count, publish=False):
        history = []
        for step in range(count):
            check_cancel()
            obs = observation(current)
            obs["validation"] = metrics(current)
            obs["recent_decisions"] = [{"accepted": row["accepted"], "reason": row["reason"]} for row in history[-3:]]
            if publish:
                (session.folder / f"observation_{step:03}.json").write_text(
                    json.dumps(obs, indent=2), encoding="utf-8")
                session.emit("inspect", step=step + 1, message="Bounded model inspection ready")
            if provider is None:
                # Explicit fixture heuristic. Odd steps reverse the last suggestion.
                delta = -min(.5, config.max_delta) if step % 2 == 0 else min(.5, config.max_delta)
                proposal = {"reason": "Guided fixture heuristic: lower bias, then test reversal",
                            "edits": [{"parameter": "2.bias", "index": 0, "delta": delta}]}
            else:
                proposal = provider.propose(obs)
            entry = {"step": step + 1, "proposal": proposal, "accepted": False}
            if publish:
                session.emit("propose", step=step + 1, proposal=proposal, message="Candidate proposed")
            if hasattr(provider, "metadata"):
                entry["provider_metadata"] = dict(provider.metadata)
            before = metrics(current)
            try:
                # Adapter lock spans validation, staging, evaluation and commit/rollback.
                with current.lock:
                    staged = current.validate(proposal)
                    changes = [{"parameter": key[0], "index": key[1], "before": old.item(),
                                "candidate": new.item()} for key, old, new, _ in staged]
                    if publish:
                        session.emit("validate", step=step + 1, changes=changes,
                                     undo_bytes=sum(old.numel() * old.element_size()
                                                    for _, old, _, _ in staged),
                                     message="Schema, numerical and cumulative limits passed")
                    with current.transaction(proposal) as transaction:
                        candidate = metrics(current)
                        if publish:
                            session.emit("evaluate", step=step + 1, candidate=candidate,
                                         message="Temporary candidate evaluated")
                        check_cancel()  # Exceptions here roll back the pending candidate.
                        entry["accepted"] = (candidate["loss"] < before["loss"] - 1e-9
                                             and candidate["accuracy"] >= before["accuracy"])
                        if entry["accepted"]:
                            transaction.commit()
                    entry.update(before=before, candidate=candidate,
                                 retained=candidate if entry["accepted"] else before,
                                 changes=changes,
                                 reason="Validation improved" if entry["accepted"]
                                 else "Loss did not improve or accuracy decreased; restored weights")
            except ValueError as exc:
                entry.update(before=before, retained=before, error=str(exc), reason=str(exc))
            history.append(entry)
            if publish:
                session.emit("accept" if entry["accepted"] else "rollback", **entry,
                             retained_origin_bytes=sum(p.numel() * p.element_size()
                                                       for p in current.originals.values()),
                             message=entry["reason"])
        return history

    provider = {"demo": lambda: None, "random": lambda: RandomProvider(config.seed),
                "guided": CalibrationProvider, "openai": OpenAIProvider}[config.provider]()
    history = trial(adapter, provider, config.budget, True)
    baselines = []
    for index in range(config.random_runs):
        check_cancel()
        other = TorchAdapter(model(), limits)
        rows = trial(other, RandomProvider(config.seed + 100 + index), config.budget)
        # Holdout is deferred until ALL searches have finished.
        baselines.append({"seed": config.seed + 100 + index, "history": rows,
                          "validation": metrics(other), "patch": other.patch()})
        session.emit("baseline", trial=index + 1, total=config.random_runs,
                     validation=baselines[-1]["validation"], message="Random baseline trial finished")
    check_cancel()
    # Replay each small sparse patch into a fresh fixture for final-only holdout.
    for baseline in baselines:
        other = TorchAdapter(model(), limits)
        if baseline["patch"]["edits"]:
            # Accepted patch may contain more edits than a single proposal permits.
            for edit in baseline["patch"]["edits"]:
                with other.transaction({"reason": "Replay retained patch", "edits": [edit]}) as tx:
                    tx.commit()
        baseline["holdout"] = metrics(other, holdout)
    torch.save(adapter.model.state_dict(), session.folder / "model.pt")
    return {"config": asdict(config), "environment": {"python": platform.python_version(),
            "torch": torch.__version__}, "fixture_sha256": digest.hexdigest(),
            "provider_disclosure": {"demo": "Scripted XOR demonstration", "guided": "Validation output-bias calibration; no backpropagation", "random": "Random visible coordinates", "openai": "Hosted LLM; validation observations sent to OpenAI"}[config.provider],
            "task": {"name": task.name, "validation_samples": len(validation[0]),
                     "holdout_samples": len(holdout[0]),
                     "holdout_start": holdout_start if config.model_id != "xor" else None},
            "initial": initial, "retained": metrics(adapter), "history": history,
            "patch": adapter.patch(), "holdout": metrics(adapter, holdout),
            "unedited_holdout": metrics(TorchAdapter(model(), limits), holdout),
            "random_baselines": baselines, "limits": asdict(limits)}
