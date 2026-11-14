"""Provider boundaries: all HTTP responses are mocked; no billable requests."""

import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from weightforge.providers import CalibrationProvider, OpenAIProvider
from weightforge.sessions import RunConfig


class ProviderTests(unittest.TestCase):
    def test_calibration_direction_and_limit(self):
        proposal = CalibrationProvider().propose({"max_delta": .2, "diagnostics": {
            "output_bias": "3.bias", "mean_target_minus_probability": [.1, -.7, .3]}})
        self.assertEqual(proposal["edits"], [{"parameter": "3.bias", "index": 1, "delta": -.2}])

    def test_explicit_consent(self):
        for consent in (False, "yes", 1):
            with self.assertRaises(ValueError):
                RunConfig(provider="openai", llm_consent=consent)
        self.assertEqual(RunConfig(provider="openai", llm_consent=True).provider, "openai")

    def provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "WEIGHTFORGE_LLM_MODEL": "test-model"}):
            return OpenAIProvider()

    def test_request_and_provenance(self):
        proposal = {"reason": "test", "edits": [{"parameter": "2.bias", "index": 0, "delta": -.1}]}
        payload = {"choices": [{"message": {"content": json.dumps(proposal)}}],
                   "id": "test-id", "model": "test-model", "usage": {"total_tokens": 100}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            provider = self.provider()
            self.assertEqual(provider.propose({"validation": {"loss": .7}}), proposal)
        request = call.call_args.args[0]
        self.assertEqual(call.call_args.kwargs["timeout"], 30)
        body = json.loads(request.data)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertNotIn("test-secret", request.data.decode())
        self.assertNotIn("holdout", request.data.decode())
        self.assertEqual(provider.metadata["request_id"], "test-id")

    def test_context_limit_prevents_network(self):
        with patch("urllib.request.urlopen") as call, self.assertRaises(ValueError):
            self.provider().propose({"text": "x" * 17000})
        call.assert_not_called()

    def test_response_limits_and_refusal(self):
        for raw in (b"x" * 65537, b'{"choices":[{"message":{"refusal":"no"}}]}'):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(raw)), self.assertRaises(ValueError):
                self.provider().propose({})

    def test_transport_failure_has_no_fallback(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("secret")) as call:
            with self.assertRaisesRegex(RuntimeError, "LLM request failed") as error:
                self.provider().propose({})
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
