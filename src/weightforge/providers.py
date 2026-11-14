"""Proposers produce untrusted JSON, never executable code or filesystem tools."""

import json
import random
from pathlib import Path
from typing import Protocol


class Provider(Protocol):
    def propose(self, observation: dict) -> dict: ...


class RandomProvider:
    """Uniform scalar coordinate, O(parameter tensors) memory, one edit per call."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def propose(self, observation):
        rows = observation["inspection"]["parameters"]
        if not rows:
            raise ValueError("No visible parameters")
        # Same visible coordinate space as manual/LLM providers.
        offset = self.rng.randrange(sum(len(p["samples"]) for p in rows))
        for p in rows:
            if offset < len(p["samples"]):
                return {
                    "reason": "Random visible coordinate baseline",
                    "edits": [
                        {
                            "parameter": p["name"],
                            "index": offset,
                            "delta": self.rng.uniform(
                                -observation["max_delta"], observation["max_delta"]
                            ),
                        }
                    ],
                }
            offset -= len(p["samples"])
        raise ValueError("No visible coordinates")


class ManualLLMProvider:
    """Optional offline LLM bridge. User supplies one JSON file per proposal.

    No network, API key, paid calls, retries, or fallback. Only observation.json
    should be sent to an external LLM. Output is always locally revalidated.
    """

    def __init__(self, files):
        self.files = iter(files)

    def propose(self, observation):
        try:
            path = Path(next(self.files))
        except StopIteration as exc:
            raise RuntimeError("Proposal files exhausted") from exc
        if path.stat().st_size > 16384:
            raise ValueError("Proposal file exceeds 16 KiB")
        return json.loads(path.read_text(encoding="utf-8"))
