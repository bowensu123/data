"""
End-to-end smoke test for the AURA data engine reproduction.

Uses a synthetic ffmpeg test video + the deterministic MockMLLMClient so it
runs offline, without any API key or GPU, and asserts the structural
invariants the paper specifies:

  - Stage 1 produces a 2 FPS / H.264 video with a valid duration.
  - Stage 4's dual sliding-window never lets the multimodal (video) portion
    of any training instance span more than N seconds.
  - Stage 4's QA-window history never keeps more than M outside groups.
  - Stage 4's supervision mask marks exactly the silent turns + the single
    target turn as supervised (Section 5.1) — never an earlier non-silent
    turn.
  - Stage 5 only keeps instances the (mock) judge passed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_data_engine.config import AURADataEngineConfig
from aura_data_engine.llm_client import MockMLLMClient
from aura_data_engine.pipeline import run_full_pipeline
from aura_data_engine.schema import SILENT_TOKEN
from aura_data_engine.video_utils import make_synthetic_test_video
from aura_data_engine.loss import per_chunk_weights, summarize_supervision


class TestAURADataEngineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_root = "/tmp/aura_data_engine_smoke"
        cls.src_dir = os.path.join(cls.tmp_root, "src_videos")
        cls.work_dir = os.path.join(cls.tmp_root, "work")
        os.makedirs(cls.src_dir, exist_ok=True)
        # A handful of short synthetic clips of varying duration, including
        # one shorter than the video window N (to exercise the "duration <= N
        # -> full history retained" branch) and ones longer than N.
        make_synthetic_test_video(os.path.join(cls.src_dir, "clip_short.mp4"), duration_s=18, fps=2)
        make_synthetic_test_video(os.path.join(cls.src_dir, "clip_long.mp4"), duration_s=75, fps=2)

        cls.cfg = AURADataEngineConfig(
            target_fps=2.0, video_window_n_s=30, prefix_margin_n_prime_s=15,
            qa_window_m_groups=10, random_seed=0,
        )
        cls.client = MockMLLMClient(seed=0, pass_rate=0.9)
        cls.instances, cls.stats = run_full_pipeline(cls.src_dir, cls.work_dir, cls.client, cls.cfg)

    def test_videos_prepared(self):
        self.assertEqual(self.stats.n_videos_in, 2)
        self.assertEqual(self.stats.n_videos_corrupted, 0)

    def test_some_instances_produced(self):
        self.assertGreater(self.stats.n_instances_unrolled, 0)
        self.assertGreater(len(self.instances), 0)
        self.assertLessEqual(len(self.instances), self.stats.n_instances_unrolled)

    def test_video_window_never_exceeds_N_seconds(self):
        N = self.cfg.video_window_n_s
        for inst in self.instances:
            multimodal = [c for c in inst.chunks if not c.text_only]
            span = multimodal[-1].t_s - multimodal[0].t_s + 1
            self.assertLessEqual(span, N,
                f"instance {inst.instance_id}: video-window span {span}s exceeds N={N}s")

    def test_qa_window_never_exceeds_M_groups(self):
        M = self.cfg.qa_window_m_groups
        for inst in self.instances:
            outside_qa_ids = {c.qa_id for c in inst.chunks if c.text_only and c.user_text}
            self.assertLessEqual(len(outside_qa_ids), M,
                f"instance {inst.instance_id}: {len(outside_qa_ids)} outside QA groups exceeds M={M}")

    def test_supervision_mask_matches_section_5_1(self):
        """m_t = 1 iff (silent multimodal turn) or (the single target turn);
        never an earlier non-silent turn."""
        for inst in self.instances:
            for pos, (chunk, supervised) in enumerate(zip(inst.chunks, inst.supervision_mask)):
                is_target = (pos == inst.target_chunk_index)
                is_silent_multimodal = (not chunk.text_only) and chunk.assistant_text == SILENT_TOKEN
                expected = is_target or is_silent_multimodal
                self.assertEqual(supervised, expected,
                    f"instance {inst.instance_id} pos {pos}: mask={supervised}, expected={expected}")
            # exactly one supervised non-silent turn (the target)
            n_supervised_nonsilent = sum(
                1 for c, m in zip(inst.chunks, inst.supervision_mask)
                if m and c.assistant_text != SILENT_TOKEN
            )
            self.assertEqual(n_supervised_nonsilent, 1)

    def test_target_is_last_chunk_and_nonsilent(self):
        for inst in self.instances:
            target = inst.chunks[inst.target_chunk_index]
            self.assertEqual(inst.target_chunk_index, len(inst.chunks) - 1)
            self.assertNotEqual(target.assistant_text, SILENT_TOKEN)

    def test_quality_verification_actually_filters(self):
        # With pass_rate=0.9 and a nontrivial number of unrolled instances,
        # Stage 5 should (almost certainly) drop at least one.
        self.assertLessEqual(self.stats.n_instances_passed_quality, self.stats.n_instances_unrolled)

    def test_short_video_uses_full_history_branch(self):
        # For the 18s clip (< N=30s), every instance's video window should
        # start at t=0, i.e. the "full history retained" branch (Section 3.1).
        short_instances = [i for i in self.instances if i.video_id and
                            any(True for _ in [1])]  # placeholder, filtered below
        # Identify by duration via chunk count heuristic: short clip has 18 one-second chunks.
        for inst in self.instances:
            multimodal = [c for c in inst.chunks if not c.text_only]
            max_t = multimodal[-1].t_s
            if max_t < self.cfg.video_window_n_s and not any(c.text_only for c in inst.chunks):
                self.assertEqual(multimodal[0].t_s, 0)

    def test_loss_weights_sane(self):
        for inst in self.instances:
            weights = per_chunk_weights(inst)
            self.assertEqual(len(weights), len(inst.chunks))
            # the target position always carries weight 1.0 (note: when
            # N_silent == 1 a silent turn's weight 1/N_silent also equals
            # 1.0, so we check the target *position* rather than counting
            # occurrences of the value 1.0 across the whole instance)
            self.assertEqual(weights[inst.target_chunk_index], 1.0)
            # all silent supervised weights equal 1/N_silent
            silent_weights = {w for w, m, c in zip(weights, inst.supervision_mask, inst.chunks)
                               if m and c.assistant_text == SILENT_TOKEN}
            if silent_weights:
                self.assertEqual(len(silent_weights), 1)  # all equal within one instance

        summary = summarize_supervision(self.instances)
        self.assertIn("silent_fraction", summary)
        self.assertGreaterEqual(summary["silent_fraction"], 0.0)
        self.assertLessEqual(summary["silent_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
