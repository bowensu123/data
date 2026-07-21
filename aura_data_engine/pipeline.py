"""
Top-level orchestrator for the Coarse-to-Fine Streaming Data Engine
(Section 4, Figure 3): Video Preparation -> QA Synthesis -> QA Refinement
-> Streaming Structuring -> Quality Verification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import AURADataEngineConfig
from .llm_client import MLLMClient
from .schema import VideoRecord, TrainingInstance, dump_jsonl
from . import stage1_video_preparation as s1
from . import stage2_qa_synthesis as s2
from . import stage3_qa_refinement as s3
from . import stage4_streaming_structuring as s4
from . import stage5_quality_verification as s5

logger = logging.getLogger("aura_data_engine")


@dataclass
class PipelineStats:
    n_videos_in: int = 0
    n_videos_corrupted: int = 0
    n_scenes: int = 0
    n_rt_candidates: int = 0
    n_rt_verified: int = 0
    n_proactive_candidates: int = 0
    n_proactive_verified: int = 0
    n_multi_verified: int = 0
    n_instances_unrolled: int = 0
    n_instances_passed_quality: int = 0

    def as_dict(self) -> Dict:
        return self.__dict__


def run_pipeline_for_video(client: MLLMClient, video: VideoRecord, cfg: AURADataEngineConfig,
                            stats: PipelineStats) -> List[TrainingInstance]:
    """Runs Stages 2-5 for a single already-prepared video."""
    # Stage 2: QA Synthesis
    synth = s2.run_qa_synthesis(client, video)
    stats.n_scenes += len(synth["scenes"])
    stats.n_rt_candidates += len(synth["real_time"])
    stats.n_proactive_candidates += len(synth["proactive"])

    # Stage 3: QA Refinement
    refined_rt, refined_proactive, refined_multi = s3.run_qa_refinement(
        client, video, synth["real_time"], synth["proactive"], synth["multi_response"], cfg)
    stats.n_rt_verified += len(refined_rt)
    stats.n_proactive_verified += len(refined_proactive)
    stats.n_multi_verified += len(refined_multi)

    # Stage 4: Streaming Structuring
    instances = s4.run_streaming_structuring(video, refined_rt, refined_proactive, refined_multi, cfg)
    stats.n_instances_unrolled += len(instances)

    # Stage 5: Quality Verification
    passed = s5.run_quality_verification(client, instances)
    stats.n_instances_passed_quality += len(passed)

    return passed


def run_full_pipeline(src_video_dir: str, work_dir: str, client: MLLMClient,
                       cfg: Optional[AURADataEngineConfig] = None,
                       domain_map: Optional[Dict[str, str]] = None,
                       ) -> "tuple[List[TrainingInstance], PipelineStats]":
    """
    End-to-end entry point: raw videos in `src_video_dir` -> list of
    TrainingInstance objects, ready to be dumped to JSONL for training.

    `work_dir` holds intermediate artifacts (prepared videos, per-video
    logs). `client` is any MLLMClient (MockMLLMClient for a dry run,
    AnthropicMLLMClient / your own subclass for real data generation).
    """
    cfg = cfg or AURADataEngineConfig()
    stats = PipelineStats()
    prepared_dir = os.path.join(work_dir, "prepared_videos")
    os.makedirs(prepared_dir, exist_ok=True)

    # Stage 1: Video Preparation
    good_videos, bad_videos = s1.run_video_preparation(src_video_dir, prepared_dir, cfg, domain_map)
    stats.n_videos_in = len(good_videos) + len(bad_videos)
    stats.n_videos_corrupted = len(bad_videos)
    for v in bad_videos:
        logger.warning("Dropping corrupted/undecodable video %s: %s", v.src_path, v.notes)

    all_instances: List[TrainingInstance] = []
    for video in good_videos:
        try:
            instances = run_pipeline_for_video(client, video, cfg, stats)
            all_instances.extend(instances)
        except Exception:
            logger.exception("Pipeline failed for video %s (%s)", video.video_id, video.src_path)

    return all_instances, stats


def save_instances(instances: List[TrainingInstance], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dump_jsonl(instances, out_path)
