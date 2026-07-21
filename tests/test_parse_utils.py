"""Tests for defensive model-output coercion and stage-2/3 tolerance of the
kind of malformed output a small local VLM produces (the real bug that made a
live Qwen2.5-VL run yield 0 instances: q_time_s returned as a list)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_data_engine.parse_utils import as_float, as_text, as_difficulty
from aura_data_engine.schema import Difficulty, VideoRecord
from aura_data_engine.llm_client import MockMLLMClient
from aura_data_engine import stage2_qa_synthesis as s2


class TestAsFloat(unittest.TestCase):
    def test_number(self):
        self.assertEqual(as_float(5), 5.0)
        self.assertEqual(as_float(5.5), 5.5)

    def test_list_the_actual_bug(self):
        # Qwen2.5-VL returned q_time_s as a list -> must not crash, take first.
        self.assertEqual(as_float([5.0]), 5.0)
        self.assertEqual(as_float([7, 9]), 7.0)

    def test_numeric_string_and_units(self):
        self.assertEqual(as_float("5"), 5.0)
        self.assertEqual(as_float("5s"), 5.0)
        self.assertEqual(as_float("5 seconds"), 5.0)

    def test_timecode(self):
        self.assertEqual(as_float("01:30"), 90.0)
        self.assertEqual(as_float("00:01:05"), 65.0)

    def test_dict(self):
        self.assertEqual(as_float({"seconds": 12}), 12.0)

    def test_uncoercible_returns_default(self):
        self.assertIsNone(as_float("soon"))
        self.assertIsNone(as_float(None))
        self.assertIsNone(as_float(True))   # bool is not a timestamp
        self.assertEqual(as_float([], default=-1.0), -1.0)


class TestAsText(unittest.TestCase):
    def test_str_and_list(self):
        self.assertEqual(as_text("hi"), "hi")
        self.assertEqual(as_text(["a", "b"]), "a b")

    def test_dict_and_default(self):
        self.assertEqual(as_text({"question": "what?"}), "what?")
        self.assertEqual(as_text(None), "")
        self.assertEqual(as_text([], default="x"), "x")


class TestAsDifficulty(unittest.TestCase):
    def test_exact_and_keyword_and_default(self):
        self.assertEqual(as_difficulty("perception"), Difficulty.L1_PERCEPTION)
        self.assertEqual(as_difficulty("Advanced Reasoning"), Difficulty.L5_ADVANCED_REASONING)
        self.assertEqual(as_difficulty("identify the object"), Difficulty.L2_RECOGNITION)
        self.assertEqual(as_difficulty("???"), Difficulty.L2_RECOGNITION)  # default


class _MalformedClient(MockMLLMClient):
    """Mimics a small VLM emitting off-schema fields."""
    def segment_scenes(self, video_path, fps, duration_s):
        return [
            {"start_s": [0], "end_s": "10s", "description": ["a", "scene"]},  # coercible
            {"start_s": "bad", "end_s": "worse", "description": "x"},          # dropped
            {"start_s": 10.0, "end_s": 20.0, "description": "ok"},
        ]

    def generate_multi_candidate_questions(self, video_path, clip_start, clip_end, scene_description, max_questions):
        # q_time_s as a list — the exact shape that crashed the live run
        return [{"question": "how many cars so far?", "q_time_s": [clip_start + 1]}]


class TestStageToleratesMalformed(unittest.TestCase):
    def setUp(self):
        self.client = _MalformedClient(seed=1)
        self.video = VideoRecord(video_id="v1", src_path="x", prepared_path="x",
                                  fps=2.0, duration_s=20.0)

    def test_segment_scenes_coerces_and_drops(self):
        scenes = s2.segment_scenes(self.client, self.video)
        # first scene coerced ([0]->0.0, "10s"->10.0), second dropped, third kept
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0].start_s, 0.0)
        self.assertEqual(scenes[0].end_s, 10.0)
        self.assertEqual(scenes[0].description, "a scene")

    def test_multi_response_does_not_crash_on_list_qtime(self):
        scenes = s2.segment_scenes(self.client, self.video)
        # Must not raise TypeError('float() argument ... not list') — the live bug.
        out = s2.synthesize_multi_response_qas(self.client, self.video, scenes)
        self.assertIsInstance(out, list)


if __name__ == "__main__":
    unittest.main()
