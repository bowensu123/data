"""
Backend for the aura_viz platform.

A dependency-light server (Python stdlib http.server + OpenCV for frame
extraction) that loads the pipeline's `training_instances.jsonl` and exposes:

  GET /                         -> the single-page frontend
  GET /static/<file>            -> frontend assets
  GET /api/summary              -> dataset-level aggregates + filter options
  GET /api/instances?...        -> filtered list of instance summaries
  GET /api/instance/<id>        -> one instance, with per-chunk weights/flags
  GET /api/frame?video=&t=      -> JPEG frame from the prepared video at t sec

It reuses the real pipeline code (schema.TrainingInstance, loss.per_chunk_weights,
loss.summarize_supervision) so what you see matches exactly what training consumes.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

# Make the sibling aura_data_engine package importable when run from anywhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from aura_data_engine.schema import TrainingInstance, SILENT_TOKEN, load_jsonl
from aura_data_engine.loss import per_chunk_weights, summarize_supervision
from aura_data_engine.config import AURADataEngineConfig
from aura_data_engine.pipeline import PipelineStats, save_instances
from aura_data_engine.agents import build_client_from_config
from aura_data_engine.llm_client import MockMLLMClient
from aura_data_engine import stage1_video_preparation as s1
from aura_data_engine import stage2_qa_synthesis as s2
from aura_data_engine import stage3_qa_refinement as s3
from aura_data_engine import stage4_streaming_structuring as s4
from aura_data_engine import stage5_quality_verification as s5

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _configs_dir() -> str:
    """Find the configs/ dir: prefer ./configs (run-from-project or wheel install),
    fall back to the repo root next to the package (editable / unzipped)."""
    cwd_cfg = os.path.join(os.getcwd(), "configs")
    if os.path.isdir(cwd_cfg):
        return cwd_cfg
    return os.path.join(_PROJECT_ROOT, "configs")


def _int_hist(values: List[int], width: int = 5) -> List[dict]:
    """Bucket integers into [k*width, k*width+width-1] bins, return ordered list."""
    if not values:
        return []
    hi = max(values)
    n_bins = hi // width + 1
    counts = [0] * n_bins
    for v in values:
        counts[v // width] += 1
    out = []
    for k, c in enumerate(counts):
        lo = k * width
        label = f"{lo}" if width == 1 else f"{lo}-{lo + width - 1}"
        out.append({"label": label, "count": c})
    return out


def _frac_hist(values: List[float], bins: int = 10) -> List[dict]:
    """Histogram of values in [0, 1] into `bins` equal buckets."""
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int(v * bins))
        counts[idx] += 1
    return [{"label": f"{k / bins:.1f}-{(k + 1) / bins:.1f}", "count": c} for k, c in enumerate(counts)]


class Dataset:
    """Loaded instances + prepared-video index for one work-dir."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.jsonl_path = os.path.join(work_dir, "training_instances.jsonl")
        self.instances: List[TrainingInstance] = []
        self.by_id: Dict[str, TrainingInstance] = {}
        self.video_paths: Dict[str, str] = {}
        self._frame_cache: Dict[str, bytes] = {}
        self.load()

    def load(self) -> None:
        self.instances = []
        if os.path.exists(self.jsonl_path):
            for raw in load_jsonl(self.jsonl_path):
                try:
                    self.instances.append(TrainingInstance.from_dict(raw))
                except Exception:
                    continue
        self.by_id = {inst.instance_id: inst for inst in self.instances}
        self._index_videos()

    def _index_videos(self) -> None:
        self.video_paths = {}
        prep = os.path.join(self.work_dir, "prepared_videos")
        if not os.path.isdir(prep):
            return
        files = [f for f in os.listdir(prep) if f.lower().endswith((".mp4", ".mkv", ".mov"))]
        for f in files:
            stem = os.path.splitext(f)[0]
            self.video_paths[stem] = os.path.join(prep, f)
        # Also map by any video_id that is a prefix of a filename (robustness).
        for inst in self.instances:
            vid = inst.video_id
            if vid in self.video_paths:
                continue
            for f in files:
                if f.startswith(vid):
                    self.video_paths[vid] = os.path.join(prep, f)
                    break

    # ---- aggregates -------------------------------------------------------
    def summary(self) -> dict:
        by_type: Dict[str, int] = {}
        videos: Dict[str, int] = {}
        for inst in self.instances:
            by_type[inst.source_qa_type.value] = by_type.get(inst.source_qa_type.value, 0) + 1
            videos[inst.video_id] = videos.get(inst.video_id, 0) + 1
        sup = summarize_supervision(self.instances)
        return {
            "work_dir": self.work_dir,
            "jsonl_path": self.jsonl_path,
            "has_jsonl": os.path.exists(self.jsonl_path),
            "n_instances": len(self.instances),
            "n_videos": len(videos),
            "n_videos_with_frames": sum(1 for v in videos if v in self.video_paths),
            "by_type": by_type,
            "videos": sorted(videos.keys()),
            "types": sorted(by_type.keys()),
            "supervision": sup,
        }

    def _turn_count(self, inst: TrainingInstance) -> int:
        return sum(1 for c in inst.chunks if not c.is_silent)

    def _first_question(self, inst: TrainingInstance) -> str:
        for c in inst.chunks:
            if c.user_text:
                return c.user_text
        return ""

    def instance_list(self, qa_type: Optional[str], video: Optional[str],
                      query: Optional[str]) -> List[dict]:
        out = []
        q = (query or "").strip().lower()
        for inst in self.instances:
            if qa_type and inst.source_qa_type.value != qa_type:
                continue
            if video and inst.video_id != video:
                continue
            fq = self._first_question(inst)
            if q:
                hay = (fq + " " + " ".join(c.assistant_text for c in inst.chunks if not c.is_silent)).lower()
                if q not in hay:
                    continue
            out.append({
                "instance_id": inst.instance_id,
                "video_id": inst.video_id,
                "source_qa_type": inst.source_qa_type.value,
                "n_chunks": len(inst.chunks),
                "n_turns": self._turn_count(inst),
                "n_silent": inst.n_silent_supervised,
                "quality_passed": inst.quality_passed,
                "first_question": fq,
                "has_frames": inst.video_id in self.video_paths,
            })
        return out

    def instance_detail(self, instance_id: str) -> Optional[dict]:
        inst = self.by_id.get(instance_id)
        if inst is None:
            return None
        weights = per_chunk_weights(inst)
        chunks = []
        for pos, (c, m, w) in enumerate(zip(inst.chunks, inst.supervision_mask, weights)):
            chunks.append({
                "pos": pos,
                "t_s": c.t_s,
                "user_text": c.user_text,
                "assistant_text": None if c.is_silent else c.assistant_text,
                "is_silent": c.is_silent,
                "is_acknowledgment": c.is_acknowledgment,
                "text_only": c.text_only,
                "qa_type": c.qa_type.value if c.qa_type else None,
                "supervised": bool(m),
                "weight": round(w, 4),
                "is_target": pos == inst.target_chunk_index,
            })
        return {
            "instance_id": inst.instance_id,
            "video_id": inst.video_id,
            "has_frames": inst.video_id in self.video_paths,
            "source_qa_type": inst.source_qa_type.value,
            "source_qa_id": inst.source_qa_id,
            "target_chunk_index": inst.target_chunk_index,
            "n_silent_supervised": inst.n_silent_supervised,
            "quality_passed": inst.quality_passed,
            "quality_reason": inst.quality_reason,
            "chunks": chunks,
        }

    # ---- analytics --------------------------------------------------------
    def analytics(self) -> dict:
        by_type: Dict[str, int] = {}
        per_video: Dict[str, int] = {}
        chunk_counts: List[int] = []
        turn_counts: List[int] = []
        span_counts: List[int] = []
        silent_fracs: List[float] = []
        quality = {"passed": 0, "filtered": 0, "unknown": 0}
        for inst in self.instances:
            by_type[inst.source_qa_type.value] = by_type.get(inst.source_qa_type.value, 0) + 1
            per_video[inst.video_id] = per_video.get(inst.video_id, 0) + 1
            chunk_counts.append(len(inst.chunks))
            turn_counts.append(sum(1 for c in inst.chunks if not c.is_silent))
            mm = [c.t_s for c in inst.chunks if not c.text_only]
            span_counts.append((max(mm) - min(mm) + 1) if mm else 0)
            sup_sil = sum(1 for c, m in zip(inst.chunks, inst.supervision_mask) if m and c.is_silent)
            sup_sp = sum(1 for c, m in zip(inst.chunks, inst.supervision_mask) if m and not c.is_silent)
            silent_fracs.append(sup_sil / max(1, sup_sil + sup_sp))
            key = "passed" if inst.quality_passed is True else ("filtered" if inst.quality_passed is False else "unknown")
            quality[key] += 1
        return {
            "n_instances": len(self.instances),
            "supervision": summarize_supervision(self.instances),
            "by_type": by_type,
            "per_video": per_video,
            "quality": quality,
            "chunks_hist": _int_hist(chunk_counts, width=5),
            "turns_hist": _int_hist(turn_counts, width=2),
            "span_hist": _int_hist(span_counts, width=5),
            "silent_fraction_hist": _frac_hist(silent_fracs, bins=10),
        }

    # ---- frame extraction -------------------------------------------------
    def frame_jpeg(self, video_id: str, t_s: float, max_w: int = 360) -> Optional[bytes]:
        path = self.video_paths.get(video_id)
        if not path or not os.path.exists(path):
            return None
        key = f"{video_id}@{round(t_s, 2)}@{max_w}"
        if key in self._frame_cache:
            return self._frame_cache[key]
        try:
            import cv2
        except ImportError:
            return None
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                # retry at 0 in case the seek overshot a short clip
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok, frame = cap.read()
            if not ok:
                return None
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (max_w, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            data = buf.tobytes()
        finally:
            cap.release()
        if len(self._frame_cache) < 2000:
            self._frame_cache[key] = data
        return data


def list_configs() -> List[dict]:
    """Available run configs: the built-in 'mock' plus every configs/*.toml|json|yaml."""
    out = [{"name": "mock", "path": "mock", "desc": "offline deterministic (no VLM)"}]
    cfg_dir = _configs_dir()
    if os.path.isdir(cfg_dir):
        for p in sorted(glob.glob(os.path.join(cfg_dir, "*.toml"))
                        + glob.glob(os.path.join(cfg_dir, "*.json"))
                        + glob.glob(os.path.join(cfg_dir, "*.yaml"))):
            out.append({"name": os.path.basename(p), "path": p, "desc": ""})
    return out


def build_client_for_run(config_name: str):
    if config_name in ("mock", "", None):
        return MockMLLMClient()
    path = config_name
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_configs_dir(), os.path.basename(config_name))
    return build_client_from_config(path)


class _JobLogHandler(logging.Handler):
    def __init__(self, job: "Job"):
        super().__init__()
        self.job = job

    def emit(self, record):
        try:
            self.job.add_log(self.format(record))
        except Exception:
            pass


class Job:
    _counter = 0

    def __init__(self, src: str, work_dir: str, config_name: str, cfg_params: dict):
        Job._counter += 1
        self.id = f"job{Job._counter}"
        self.src = src
        self.work_dir = work_dir
        self.config_name = config_name
        self.cfg_params = cfg_params
        self.state = "starting"           # starting | running | done | error | cancelled
        self.stage = "queued"
        self.video_i = 0
        self.video_n = 0
        self.stats: dict = {}
        self.log: List[str] = []
        self.error: Optional[str] = None
        self.n_instances = 0
        self.started = time.time()
        self.finished: Optional[float] = None
        self._cancel = False
        self._lock = threading.Lock()

    def add_log(self, msg: str):
        with self._lock:
            self.log.append(msg)
            if len(self.log) > 400:
                self.log = self.log[-400:]

    def set_stage(self, s: str):
        with self._lock:
            self.stage = s

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id, "state": self.state, "stage": self.stage,
                "video_i": self.video_i, "video_n": self.video_n,
                "stats": dict(self.stats), "n_instances": self.n_instances,
                "src": self.src, "work_dir": self.work_dir, "config": self.config_name,
                "error": self.error, "log": list(self.log[-60:]),
                "elapsed_s": round((self.finished or time.time()) - self.started, 1),
            }


class JobManager:
    def __init__(self):
        self.current: Optional[Job] = None
        self._lock = threading.Lock()

    def start(self, src: str, work_dir: str, config_name: str, cfg_params: dict) -> Job:
        with self._lock:
            if self.current and self.current.state in ("starting", "running"):
                raise RuntimeError("a pipeline run is already in progress")
            job = Job(src, work_dir, config_name, cfg_params)
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job):
        handler = _JobLogHandler(job)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pipe_logger = logging.getLogger("aura_data_engine")
        pipe_logger.addHandler(handler)
        try:
            job.state = "running"
            job.add_log(f"building client from config '{job.config_name}'")
            client = build_client_for_run(job.config_name)
            p = job.cfg_params or {}
            cfg = AURADataEngineConfig(
                target_fps=float(p.get("fps", 2.0)),
                video_window_n_s=int(p.get("video_window_n", 30)),
                qa_window_m_groups=int(p.get("qa_window_m", 10)),
                random_seed=int(p.get("seed", 0)),
            )
            prepared_dir = os.path.join(job.work_dir, "prepared_videos")
            os.makedirs(prepared_dir, exist_ok=True)

            job.set_stage("Stage 1 — preparing videos")
            good, bad = s1.run_video_preparation(job.src, prepared_dir, cfg)
            job.video_n = len(good)
            job.add_log(f"Stage 1: {len(good)} prepared, {len(bad)} dropped")

            stats = PipelineStats()
            stats.n_videos_in = len(good) + len(bad)
            stats.n_videos_corrupted = len(bad)
            all_instances = []
            for i, video in enumerate(good):
                if job._cancel:
                    job.state = "cancelled"
                    job.set_stage("cancelled")
                    return
                job.video_i = i + 1
                tag = f"video {i + 1}/{len(good)}"
                try:
                    job.set_stage(f"{tag} — Stage 2 QA synthesis")
                    synth = s2.run_qa_synthesis(client, video)
                    stats.n_scenes += len(synth["scenes"])
                    stats.n_rt_candidates += len(synth["real_time"])
                    stats.n_proactive_candidates += len(synth["proactive"])

                    job.set_stage(f"{tag} — Stage 3 refinement")
                    rr, rp, rm = s3.run_qa_refinement(
                        client, video, synth["real_time"], synth["proactive"], synth["multi_response"], cfg)
                    stats.n_rt_verified += len(rr)
                    stats.n_proactive_verified += len(rp)
                    stats.n_multi_verified += len(rm)

                    job.set_stage(f"{tag} — Stage 4 streaming structuring")
                    insts = s4.run_streaming_structuring(video, rr, rp, rm, cfg)
                    stats.n_instances_unrolled += len(insts)

                    job.set_stage(f"{tag} — Stage 5 quality verification")
                    passed = s5.run_quality_verification(client, insts)
                    stats.n_instances_passed_quality += len(passed)
                    all_instances.extend(passed)
                    job.add_log(f"{tag}: {len(passed)} instances kept (total {len(all_instances)})")
                except Exception as e:  # noqa: BLE001 - one bad video must not kill the run
                    job.add_log(f"{tag}: FAILED — {e}")
                with job._lock:
                    job.stats = stats.as_dict()

            out_path = os.path.join(job.work_dir, "training_instances.jsonl")
            save_instances(all_instances, out_path)
            job.n_instances = len(all_instances)
            job.add_log(f"wrote {len(all_instances)} instances -> {out_path}")
            job.set_stage("done")
            job.state = "done"
        except Exception as e:  # noqa: BLE001
            job.error = str(e)
            job.state = "error"
            job.add_log(f"ERROR: {e}")
            job.set_stage("error")
        finally:
            job.finished = time.time()
            pipe_logger.removeHandler(handler)


DATASET: Optional[Dataset] = None
JOBS = JobManager()


class Handler(BaseHTTPRequestHandler):
    server_version = "aura_viz/0.1"

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/summary":
                return self._json(DATASET.summary())
            if path == "/api/analytics":
                return self._json(DATASET.analytics())
            if path == "/api/configs":
                return self._json(list_configs())
            if path == "/api/job":
                return self._json(JOBS.current.snapshot() if JOBS.current else {"state": "idle"})
            if path == "/api/instances":
                return self._json(DATASET.instance_list(
                    qs.get("type", [None])[0], qs.get("video", [None])[0], qs.get("q", [None])[0]))
            if path.startswith("/api/instance/"):
                detail = DATASET.instance_detail(urllib.parse.unquote(path[len("/api/instance/"):]))
                return self._json(detail, 200 if detail else 404) if detail else self._json({"error": "not found"}, 404)
            if path == "/api/frame":
                vid = qs.get("video", [""])[0]
                t = float(qs.get("t", ["0"])[0])
                data = DATASET.frame_jpeg(vid, t)
                if data is None:
                    return self._send(204, b"", "image/jpeg")
                return self._send(200, data, "image/jpeg", cache=True)
            if path == "/api/reload":
                DATASET.load()
                return self._json({"reloaded": True, "n_instances": len(DATASET.instances)})
            return self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 - never crash the dev server
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        global DATASET
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8") or "{}")
        except Exception:
            return self._json({"error": "invalid JSON body"}, 400)
        try:
            if path == "/api/run":
                src = (data.get("src") or "").strip()
                work_dir = (data.get("work_dir") or "").strip()
                config = data.get("config") or "mock"
                params = data.get("params") or {}
                if not src or not os.path.isdir(src):
                    return self._json({"error": f"source video dir not found: {src!r}"}, 400)
                if not work_dir:
                    return self._json({"error": "work_dir is required"}, 400)
                job = JOBS.start(src, work_dir, config, params)
                return self._json(job.snapshot())
            if path == "/api/stop":
                if JOBS.current and JOBS.current.state in ("starting", "running"):
                    JOBS.current._cancel = True
                    return self._json({"stopping": True})
                return self._json({"error": "no run in progress"}, 400)
            if path == "/api/load":
                wd = (data.get("work_dir") or "").strip()
                if not os.path.exists(os.path.join(wd, "training_instances.jsonl")):
                    return self._json({"error": f"no training_instances.jsonl in {wd!r}"}, 400)
                DATASET = Dataset(wd)
                return self._json(DATASET.summary())
            return self._json({"error": "not found"}, 404)
        except RuntimeError as e:
            return self._json({"error": str(e)}, 409)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def _static(self, rel: str):
        rel = rel.split("?")[0].lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            return self._json({"error": "not found"}, 404)
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as f:
            return self._send(200, f.read(), ctype)


def serve(work_dir: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    global DATASET
    DATASET = Dataset(work_dir)
    s = DATASET.summary()
    print(f"aura_viz: loaded {s['n_instances']} instances from {DATASET.jsonl_path}")
    if not s["has_jsonl"]:
        print(f"  (no training_instances.jsonl found in {work_dir} — run the pipeline first)")
    print(f"  videos with frames on disk: {s['n_videos_with_frames']}/{s['n_videos']}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"aura_viz: serving at {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\naura_viz: stopped")
        httpd.server_close()
