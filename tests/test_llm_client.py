"""Regression tests for llm_client JSON extraction and the DashScope /
OpenAI-compatible client's content-format fallback."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_data_engine.llm_client import _extract_json, DashScopeMLLMClient


class _Boom(Exception):
    """Fake OpenAI-style APIStatusError carrying a status_code."""
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestExtractJson(unittest.TestCase):
    def test_object_with_inner_array_and_preamble(self):
        # The bug: '['-first recovery returned the inner [1, 2] instead of the
        # object. Must now return the object.
        got = _extract_json('Here is the result: {"pass": false, "reason": "boxes [1, 2] missing"}')
        self.assertEqual(got, {"pass": False, "reason": "boxes [1, 2] missing"})

    def test_object_with_suffix(self):
        got = _extract_json('Sure! {"pass": true, "reason": "coords [10, 20, 30, 40]"} Hope this helps.')
        self.assertEqual(got, {"pass": True, "reason": "coords [10, 20, 30, 40]"})

    def test_plain_array_still_works(self):
        self.assertEqual(_extract_json('```json\n[{"a": 1}, {"a": 2}]\n```'), [{"a": 1}, {"a": 2}])

    def test_clean_object(self):
        self.assertEqual(_extract_json('{"pass": true}'), {"pass": True})


class TestVideoToImageFallback(unittest.TestCase):
    def _client(self, max_retries):
        c = DashScopeMLLMClient(api_key="test-key", max_retries=max_retries)
        c._sample_frames_b64 = lambda *a, **k: ["AAAA", "BBBB"]  # >=2 -> 'video' block
        return c

    def test_fallback_retries_even_with_max_retries_1(self):
        """Finding #1: with max_retries=1, a 400 rejecting the 'video' block must
        still trigger an actual image_url retry, not re-raise the 400."""
        c = self._client(max_retries=1)
        seen = []

        def fake_create(content):
            seen.append([b["type"] for b in content])
            if any(b["type"] == "video" for b in content):
                raise _Boom(400)
            return '{"ok": true}'

        c._create = fake_create
        out = c._call("prompt", video_path="x.mp4")
        self.assertEqual(out, '{"ok": true}')
        self.assertEqual(len(seen), 2)                 # video rejected, then image_url
        self.assertIn("video", seen[0])
        self.assertIn("image_url", seen[1])
        self.assertEqual(c.frame_format, "image_url")  # sticky for the rest of the run

    def test_403_is_not_retried_and_gives_hint(self):
        c = self._client(max_retries=4)
        n = {"calls": 0}

        def fake_create(content):
            n["calls"] += 1
            raise _Boom(403)

        c._create = fake_create
        with self.assertRaises(RuntimeError) as ctx:
            c._call("prompt", video_path="x.mp4")
        self.assertEqual(n["calls"], 1)               # no retries on auth failure
        self.assertIn("AccessDenied", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
