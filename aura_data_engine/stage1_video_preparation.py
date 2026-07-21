"""
Stage 1 — Video Preparation (Section 4.1).

"We collect high-quality videos from public internet sources covering a
broad range of categories ... each video is resampled to a fixed frame
rate of 2 FPS ... and re-encoded in H.264 format ... while mitigating
issues caused by corrupted or undecodable frames."

This stage takes a directory of raw source videos (optionally with a
sidecar `domain` label per file) and produces standardized VideoRecord
objects with a prepared 2 FPS / H.264 copy on disk, filtering out anything
ffmpeg/ffprobe cannot decode.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .config import AURADataEngineConfig
from .schema import VideoRecord, new_id
from . import video_utils

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def discover_videos(src_dir: str, domain_map: Optional[Dict[str, str]] = None) -> List[str]:
    """List candidate video files under `src_dir`. `domain_map` optionally maps
    filename -> domain label (sports/vlogs/documentaries/... per Fig. 5)."""
    paths = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def prepare_video(src_path: str, out_dir: str, cfg: AURADataEngineConfig,
                   domain: str = "unknown") -> VideoRecord:
    """Run one source video through resample+re-encode, returning a VideoRecord.

    On decode failure the returned record has `corrupted=True` and should be
    dropped before Stage 2.
    """
    video_id = new_id("vid")
    dst_path = os.path.join(out_dir, f"{video_id}.mp4")
    rec = VideoRecord(video_id=video_id, src_path=src_path, domain=domain, fps=cfg.target_fps)

    try:
        src_info = video_utils.ffprobe_info(src_path)
    except RuntimeError as e:
        rec.corrupted = True
        rec.notes = f"ffprobe failed on source: {e}"
        return rec

    ok, msg = video_utils.resample_and_reencode(
        src_path, dst_path, target_fps=cfg.target_fps, codec=cfg.video_codec)
    if not ok:
        rec.corrupted = True
        rec.notes = f"re-encode failed: {msg}"
        return rec

    try:
        dst_info = video_utils.ffprobe_info(dst_path)
    except RuntimeError as e:
        rec.corrupted = True
        rec.notes = f"ffprobe failed on prepared output: {e}"
        return rec

    rec.prepared_path = dst_path
    rec.duration_s = dst_info["duration_s"]
    rec.width = dst_info["width"]
    rec.height = dst_info["height"]
    rec.fps = dst_info["fps"] or cfg.target_fps
    return rec


def run_video_preparation(src_dir: str, out_dir: str, cfg: AURADataEngineConfig,
                           domain_map: Optional[Dict[str, str]] = None
                           ) -> "tuple[List[VideoRecord], List[VideoRecord]]":
    """Batch entry point for Stage 1. Returns (good_records, corrupted_records)."""
    domain_map = domain_map or {}
    records = []
    for src_path in discover_videos(src_dir, domain_map):
        domain = domain_map.get(os.path.basename(src_path), "unknown")
        rec = prepare_video(src_path, out_dir, cfg, domain=domain)
        records.append(rec)
    return [r for r in records if not r.corrupted], [r for r in records if r.corrupted]
