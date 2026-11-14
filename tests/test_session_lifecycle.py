"""Standard-library lifecycle checks; runnable without tensor/API dependencies.

Run: python -m unittest discover -s tests -p test_session_lifecycle.py -v
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from weightforge.sessions import Cancelled, RunConfig, SessionManager


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def manager(self, runner, capacity=20):
        manager = SessionManager(Path(self.temp.name), runner, capacity)
        self.addCleanup(manager.close)
        return manager

    def finish(self, session):
        deadline = time.monotonic() + 5
        while session.snapshot()["status"] not in {"completed", "failed", "cancelled"}:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        return session.snapshot()

    def test_persistence_and_event_replay(self):
        def runner(session):
            session.emit("started", message="test runner")
            return {"seed": session.config.seed}

        manager = self.manager(runner)
        session = manager.create(RunConfig(seed=5))
        self.assertEqual(self.finish(session)["status"], "completed")
        self.assertEqual(json.loads((session.folder / "results.json").read_text()), {"seed": 5})
        events, done = session.wait_events(1)
        self.assertTrue(done)
        self.assertEqual([row["stage"] for row in events], ["started", "completed"])
        self.assertEqual([row["id"] for row in session.events], [1, 2, 3])

    def test_queue_cancellation_and_capacity(self):
        gate = threading.Event()
        self.addCleanup(gate.set)

        def runner(session):
            gate.wait(3)
            if session.cancel.is_set():
                raise Cancelled()
            return {"identity": session.id}

        manager = self.manager(runner)
        one = manager.create(RunConfig(seed=1))
        two = manager.create(RunConfig(seed=2))
        two.cancel.set()
        with self.assertRaises(ValueError):
            manager.create(RunConfig())
        gate.set()
        self.assertEqual(self.finish(one)["status"], "completed")
        self.assertEqual(self.finish(two)["status"], "cancelled")
        self.assertIsNone(two.result)
        self.assertNotEqual(one.folder, two.folder)

    def test_failure_is_visible(self):
        def runner(session):
            raise RuntimeError("test inference failure")

        session = self.manager(runner).create(RunConfig())
        self.assertEqual(self.finish(session)["status"], "failed")
        self.assertEqual(session.events[-1]["message"], "test inference failure")

    def test_bounded_registry_evicts_terminal_sessions(self):
        manager = self.manager(lambda session: {}, capacity=1)
        one = manager.create(RunConfig())
        self.finish(one)
        two = manager.create(RunConfig())
        self.finish(two)
        with self.assertRaises(KeyError):
            manager.get(one.id)
        self.assertIs(manager.get(two.id), two)

    def test_strict_config(self):
        invalid = [{"budget": True}, {"budget": 51}, {"provider": "llm"}, {"max_delta": float("nan")},
                   {"random_runs": 0}, {"seed": -1}, {"max_changed": 10}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                RunConfig(**config)


if __name__ == "__main__":
    unittest.main()
