"""ffmpeg/opencv helpers used by Stage 1 (Video Preparation, Section 4.1).

Prefers a system ``ffmpeg``/``ffprobe`` (faithful 2 FPS + H.264 re-encode as
described in Section 4.1). When those binaries are not on PATH, it falls back
to an OpenCV implementation using the FFmpeg libraries bundled inside
``opencv-python[-headless]`` — so the pipeline runs end-to-end with no system
install. The OpenCV encoder is best-effort H.264 (fourcc ``avc1``) and drops
to ``mp4v`` if no H.264 writer is available; either way the output is a
re-sampled, OpenCV-readable clip, which is all the downstream frame sampler
needs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Tuple

# Quiet the noisy (but harmless) FFmpeg/OpenH264 stderr chatter that OpenCV's
# bundled backend emits when it probes an H.264 (avc1) VideoWriter and falls
# back — set before cv2 first initializes its FFmpeg plugin. AV_LOG_QUIET = -8.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

logger = logging.getLogger("aura_data_engine.video_utils")


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


# --------------------------------------------------------------------------- #
# ffprobe / metadata
# --------------------------------------------------------------------------- #

def _opencv_probe(path: str) -> dict:
    """Read width/height/fps/duration via OpenCV (ffprobe fallback)."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if width == 0 or height == 0:
        raise RuntimeError(f"OpenCV read zero dimensions for {path} (undecodable?)")
    duration = (n_frames / fps) if fps else 0.0
    return {"width": width, "height": height, "fps": fps, "duration_s": duration}


def ffprobe_info(path: str) -> dict:
    """Return duration/fps/width/height via ffprobe, or raise on decode failure.

    Falls back to OpenCV when the ``ffprobe`` binary is unavailable.
    """
    if not _have("ffprobe"):
        return _opencv_probe(path)

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})

    def _parse_rate(r: str) -> float:
        if not r or r == "0/0":
            return 0.0
        num, _, den = r.partition("/")
        den = den or "1"
        return float(num) / float(den) if float(den) else 0.0

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": _parse_rate(stream.get("r_frame_rate", "0/0")),
        "duration_s": duration,
    }


# --------------------------------------------------------------------------- #
# resample + re-encode
# --------------------------------------------------------------------------- #

def _opencv_resample(src_path: str, dst_path: str, target_fps: float) -> Tuple[bool, str]:
    """Resample `src_path` to `target_fps` and write `dst_path` with OpenCV.

    Best-effort H.264 (fourcc 'avc1'); falls back to 'mp4v' if no H.264
    VideoWriter is available in this OpenCV build. Returns (success, message);
    on failure `success` is False so corrupted sources are filtered upstream.
    """
    import cv2
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        cap.release()
        return False, "OpenCV could not open source video"
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        return False, "OpenCV read zero dimensions (undecodable source)"
    duration = (n_frames / src_fps) if src_fps else 0.0

    # Prefer mp4v: it is always present in OpenCV's bundled build and writes
    # silently, whereas 'avc1' (true H.264) needs an openh264 runtime that is
    # often missing and emits noisy load-failure chatter when it is. The
    # codec of this intermediate, re-sampled clip does not matter — it is only
    # re-decoded to JPEG frames downstream — and a real system ffmpeg (used
    # when available) still produces genuine H.264. 'avc1' is kept as a second
    # choice for builds that do support it.
    writer = None
    used = None
    for fourcc_name in ("mp4v", "avc1"):
        w = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*fourcc_name), target_fps, (width, height))
        if w.isOpened():
            writer, used = w, fourcc_name
            break
        w.release()
    if writer is None:
        cap.release()
        return False, "no working OpenCV VideoWriter codec (tried mp4v, avc1)"

    written = 0
    try:
        if duration > 0:
            # Seek to each target timestamp — accurate for the short clips this
            # pipeline targets.
            n_out = max(1, int(round(duration * target_fps)))
            step_ms = 1000.0 / target_fps
            for i in range(n_out):
                cap.set(cv2.CAP_PROP_POS_MSEC, i * step_ms)
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        else:
            # Unknown duration: decimate the frame stream sequentially.
            keep_every = max(1, round(src_fps / target_fps)) if src_fps else 1
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % keep_every == 0:
                    writer.write(frame)
                    written += 1
                idx += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        return False, "OpenCV wrote zero frames"
    return True, f"ok (opencv fallback, fourcc={used}, {written} frames @ {target_fps} FPS)"


def resample_and_reencode(src_path: str, dst_path: str, target_fps: float = 2.0,
                           codec: str = "libx264") -> Tuple[bool, str]:
    """
    Resample to `target_fps` and re-encode with `codec` (Section 4.1: "each video is
    resampled to a fixed frame rate of 2 FPS ... and re-encoded in H.264 format").

    Uses ffmpeg when available; otherwise falls back to OpenCV (see
    `_opencv_resample`). Returns (success, message). On failure, `success` is
    False and `message` holds the error, so corrupted/undecodable sources can
    be filtered out upstream ("mitigating issues caused by corrupted or
    undecodable frames").
    """
    if not _have("ffmpeg"):
        return _opencv_resample(src_path, dst_path, target_fps)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vf", f"fps={target_fps}",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-an",  # audio is handled by the separate ASR pipeline, not the training video stream
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()[-2000:]
    return True, "ok"


def _opencv_synthetic(dst_path: str, duration_s: float, fps: float) -> None:
    """OpenCV fallback for `make_synthetic_test_video` (no ffmpeg needed)."""
    import cv2
    import numpy as np
    width, height = 320, 240
    n_frames = max(1, int(round(duration_s * fps)))

    writer = None
    for fourcc_name in ("mp4v", "avc1"):  # mp4v first: always present, no openh264 chatter
        w = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
        if w.isOpened():
            writer = w
            break
        w.release()
    if writer is None:
        raise RuntimeError("no working OpenCV VideoWriter codec for synthetic clip (tried mp4v, avc1)")

    try:
        for i in range(n_frames):
            t = i / fps
            # A hue that drifts over time so consecutive frames differ.
            hue = int((i * 7) % 180)
            hsv = np.full((height, width, 3), (hue, 200, 200), np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.putText(frame, f"t={t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def make_synthetic_test_video(dst_path: str, duration_s: float = 40.0, fps: float = 2.0) -> None:
    """
    Generate a tiny synthetic clip (color field + timestamp burn-in) purely for
    exercising the pipeline end-to-end without real footage. Not used for
    anything resembling real data generation. Uses ffmpeg when available,
    else an OpenCV fallback.
    """
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    if not _have("ffmpeg"):
        _opencv_synthetic(dst_path, duration_s, fps)
        return
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={fps}:duration={duration_s}",
        "-vf", "drawtext=text='t=%{pts\\:hms}':x=10:y=10:fontcolor=white:fontsize=16:box=1:boxcolor=black@0.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        dst_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
