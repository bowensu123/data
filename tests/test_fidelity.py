"""Regression tests for AURA-fidelity fixes (2026-07-22): each locks in an
algorithmic rule tightened to match the paper."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_data_engine.schema import ProactiveQA, Scene, VideoRecord
from aura_data_engine.llm_client import OpenAICompatibleMLLMClient, MockMLLMClient
from aura_data_engine import stage2_qa_synthesis as s2


class TestProactiveStrictTiming(unittest.TestCase):
    """§4.2: the question timestamp *precedes* the answer timestamp."""
    def test_equal_timestamps_rejected(self):
        with self.assertRaises(AssertionError):
            ProactiveQA(qa_id="p", video_id="v", question="q", answer="a",
                        q_time_s=5.0, a_time_s=5.0)

    def test_strictly_later_ok(self):
        ProactiveQA(qa_id="p", video_id="v", question="q", answer="a",
                    q_time_s=5.0, a_time_s=6.0)  # must not raise


class TestStage5IsVisual(unittest.TestCase):
    """§4.5: the judge sees the retained video window's frames, not text alone."""
    def _client(self):
        c = OpenAICompatibleMLLMClient(model="dummy")
        c._sample_frames_b64 = lambda *a, **k: ["AAAA", "BBBB"]  # 2 fake frames
        self.captured = {}
        def fake_create(content):
            self.captured["types"] = [b["type"] for b in content]
            return '{"pass": true, "reason": "ok"}'
        c._create = fake_create
        return c

    def test_rt_verify_attaches_frames(self):
        c = self._client()
        c.quality_verify_rt("v.mp4", 2, 10, "hist", "q", "a", 10)
        self.assertTrue(any(t in ("image_url", "video") for t in self.captured["types"]))

    def test_proactive_multi_verify_attaches_frames(self):
        c = self._client()
        c.quality_verify_proactive_multi("v.mp4", 2, 10, "hist", "q", "a", 10, 3)
        self.assertTrue(any(t in ("image_url", "video") for t in self.captured["types"]))


class _DupTimestampClient(MockMLLMClient):
    """Emits multi-response answers where two share the same chunk-second."""
    def generate_multi_candidate_questions(self, video_path, clip_start, clip_end, scene_description, max_questions):
        return [{"question": "how many so far?", "q_time_s": clip_start + 0.5}]

    def check_multi_answerable(self, *a, **k):
        return {"pass": True, "reason": "ok"}

    def generate_multi_answers(self, video_path, clip_start, clip_end, question, q_time):
        return [{"text": "one", "timestamp_s": q_time + 1.0},
                {"text": "one-dup", "timestamp_s": q_time + 1.2},   # same rounded second
                {"text": "two", "timestamp_s": q_time + 3.0}]

    def verify_multi_answer(self, *a, **k):
        return {"pass": True, "reason": "ok"}


class TestMultiResponseDistinctTimestamps(unittest.TestCase):
    """§4.2: multiple valid answers at *different* timestamps."""
    def test_duplicate_timestamps_deduped(self):
        client = _DupTimestampClient(seed=0)
        video = VideoRecord(video_id="v", src_path="x", prepared_path="x", duration_s=12.0)
        scenes = [Scene(scene_id="s", video_id="v", start_s=0.0, end_s=10.0, description="d")]
        out = s2.synthesize_multi_response_qas(client, video, scenes)
        self.assertEqual(len(out), 1)
        ts = [round(a.timestamp_s) for a in out[0].answers]
        self.assertEqual(len(ts), len(set(ts)), "answers must have distinct timestamps")
        self.assertGreaterEqual(len(ts), 2)


if __name__ == "__main__":
    unittest.main()
