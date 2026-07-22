"""
MLLM / LLM client abstraction.

The paper uses an unspecified MLLM for scene segmentation, QA synthesis,
refinement, and verification (Section 4), and a plain LLM for question
rewriting (Section 4.3). This module defines one interface, `MLLMClient`,
that every pipeline stage codes against, plus implementations:

  * `MockMLLMClient`   - deterministic, offline, seeded-random stand-in.
                          Lets you run and unit-test the *data engine logic*
                          (windowing, unrolling, masking, filtering ratios)
                          without any GPU or API key. This is what the
                          bundled smoke test uses.

  * `AnthropicMLLMClient` - a real client that samples frames from the
                          prepared video with OpenCV and calls the Claude
                          API (vision + text) with the prompts in prompts.py.

  * `DashScopeMLLMClient` - a real client for Aliyun Bailian (百炼) / DashScope
                          Qwen-VL models, via the OpenAI-compatible endpoint.
                          Same frame-sampling path; this is the backend to use
                          with a Bailian API key.

The two real backends share every Stage 2/3/5 method through the
`_FrameSampledMLLMClient` base class and differ only in their `_call`
primitive (which chat API they hit and how they package frames). To add
another provider (GPT-4o, a local Qwen served with vLLM, ...), subclass
`_FrameSampledMLLMClient` and implement `_call`.

All methods return already-parsed Python objects (lists/dicts), matching
the "Return ONLY a JSON ..." instruction baked into every prompt.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from . import prompts

logger = logging.getLogger("aura_data_engine.llm_client")


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM text response.

    Real MLLMs occasionally wrap JSON in markdown fences or add a preamble
    despite instructions; this strips the common cases before parsing.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fall back to grabbing the first {...} or [...] block.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Recover a JSON value embedded in prose. Prefer whichever of {...} or
        # [...] opens FIRST in the text (i.e. the outermost structure), so an
        # object response whose string fields contain a bracketed array — e.g.
        # {"pass": false, "reason": "boxes [1, 2] missing"} — is parsed as the
        # object, not mis-parsed as that inner array.
        candidates = []
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                candidates.append((start, text[start:end + 1]))
        for _, candidate in sorted(candidates, key=lambda c: c[0]):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Could not parse JSON from model output: {text[:200]!r}")


class MLLMClient(ABC):
    """Interface every pipeline stage codes against."""

    # ---- Stage 2: QA Synthesis ----
    @abstractmethod
    def segment_scenes(self, video_path: str, fps: float, duration_s: float) -> List[Dict]:
        ...

    @abstractmethod
    def generate_candidate_qas(self, video_path: str, scenes: List[Dict], max_qas: int) -> List[Dict]:
        ...

    @abstractmethod
    def verify_rt_qa(self, video_path: str, question: str, answer: str, a_time: float) -> Dict:
        ...

    @abstractmethod
    def verify_proactive_qa(self, video_path: str, question: str, answer: str,
                             q_time: float, a_time: float) -> Dict:
        ...

    @abstractmethod
    def generate_multi_candidate_questions(self, video_path: str, clip_start: float, clip_end: float,
                                            scene_description: str, max_questions: int) -> List[Dict]:
        ...

    @abstractmethod
    def check_multi_answerable(self, video_path: str, clip_start: float, clip_end: float,
                                question: str, q_time: float) -> Dict:
        ...

    @abstractmethod
    def generate_multi_answers(self, video_path: str, clip_start: float, clip_end: float,
                                question: str, q_time: float) -> List[Dict]:
        ...

    @abstractmethod
    def verify_multi_answer(self, video_path: str, question: str, answer: str, timestamp: float) -> Dict:
        ...

    # ---- Stage 3: QA Refinement ----
    @abstractmethod
    def generate_difficulty_siblings(self, video_path: str, question: str, t: float) -> List[Dict]:
        ...

    @abstractmethod
    def generate_answer_for_question(self, video_path: str, question: str, t: float) -> Dict:
        ...

    @abstractmethod
    def rewrite_question(self, question: str, template: str) -> Dict:
        ...

    # ---- Stage 5: Quality Verification ----
    # video_path + [window_start, window_end] let the judge see the retained video
    # window's actual frames (Section 4.5 requires a *visual* grounding check).
    @abstractmethod
    def quality_verify_rt(self, video_path: str, window_start: float, window_end: float, qa_history: str,
                           target_question: str, target_answer: str, target_time: float) -> Dict:
        ...

    @abstractmethod
    def quality_verify_proactive_multi(self, video_path: str, window_start: float, window_end: float,
                                        qa_history: str, target_question: str, target_answer: str,
                                        target_time: float, query_time: float) -> Dict:
        ...


# --------------------------------------------------------------------------- #
# Mock implementation - deterministic, offline
# --------------------------------------------------------------------------- #

class MockMLLMClient(MLLMClient):
    """
    Deterministic stand-in MLLM.

    Uses a seeded RNG plus lightweight video-duration/timing awareness so
    the *pipeline mechanics* (windowing, unrolling, filtering, sampling
    ratios) can be exercised and unit-tested without any real model.
    Do NOT use this to produce actual training data — its "content" is
    synthetic and unrelated to what's actually in any given video frame.
    """

    def __init__(self, seed: int = 0, pass_rate: float = 0.85):
        self.rng = random.Random(seed)
        self.pass_rate = pass_rate

    def _passes(self) -> bool:
        return self.rng.random() < self.pass_rate

    def segment_scenes(self, video_path, fps, duration_s):
        n_scenes = max(1, int(duration_s // 8))
        bounds = sorted(self.rng.sample(range(1, int(duration_s)), k=min(n_scenes - 1, max(0, int(duration_s) - 1)))) \
            if n_scenes > 1 else []
        bounds = [0.0] + [float(b) for b in bounds] + [duration_s]
        scenes = []
        for i in range(len(bounds) - 1):
            scenes.append({
                "start_s": bounds[i],
                "end_s": bounds[i + 1],
                "description": f"Synthetic scene {i}: something happens between "
                                f"{bounds[i]:.1f}s and {bounds[i + 1]:.1f}s.",
            })
        return scenes

    def generate_candidate_qas(self, video_path, scenes, max_qas):
        out = []
        for i, sc in enumerate(scenes[:max_qas]):
            mid = (sc["start_s"] + sc["end_s"]) / 2.0
            if i % 2 == 0:
                out.append({
                    "type": "real_time",
                    "question": f"What is happening around {mid:.1f}s (scene {i})?",
                    "answer": f"[synthetic] The event described in scene {i} is occurring.",
                    "q_time_s": mid,
                    "a_time_s": mid,
                })
            else:
                q_t = sc["start_s"]
                a_t = min(sc["end_s"], q_t + 3.0)
                out.append({
                    "type": "proactive",
                    "question": f"Will the event in scene {i} resolve successfully?",
                    "answer": f"[synthetic] Yes, by {a_t:.1f}s it resolves as expected.",
                    "q_time_s": q_t,
                    "a_time_s": a_t,
                })
        return out

    def verify_rt_qa(self, video_path, question, answer, a_time):
        return {"pass": self._passes(), "reason": "mock verification"}

    def verify_proactive_qa(self, video_path, question, answer, q_time, a_time):
        return {"pass": self._passes(), "reason": "mock verification"}

    def generate_multi_candidate_questions(self, video_path, clip_start, clip_end, scene_description, max_questions):
        q_t = clip_start + 0.5 * (clip_end - clip_start) * 0.2
        return [{
            "question": "How many times has the tracked event recurred so far?",
            "q_time_s": q_t,
        }]

    def check_multi_answerable(self, video_path, clip_start, clip_end, question, q_time):
        return {"pass": self._passes(), "reason": "mock check"}

    def generate_multi_answers(self, video_path, clip_start, clip_end, question, q_time):
        n = self.rng.randint(2, 4)
        answers = []
        span = clip_end - q_time
        for k in range(n):
            ts = q_time + span * (k + 1) / (n + 1)
            answers.append({"text": f"[synthetic] Update #{k + 1} at {ts:.1f}s.", "timestamp_s": ts})
        return answers

    def verify_multi_answer(self, video_path, question, answer, timestamp):
        return {"pass": self._passes(), "reason": "mock verification"}

    def generate_difficulty_siblings(self, video_path, question, t):
        levels = ["perception", "recognition", "understanding", "reasoning", "advanced_reasoning"]
        return [{"difficulty": lvl, "question": f"[{lvl}] variant of: {question}"} for lvl in levels[:4]]

    def generate_answer_for_question(self, video_path, question, t):
        return {"answer": f"[synthetic answer @ {t:.1f}s] {question[::-1][:0]}Answer to: {question}"}

    def rewrite_question(self, question, template):
        return {"question": template.format(question)}

    def quality_verify_rt(self, video_path, window_start, window_end, qa_history,
                           target_question, target_answer, target_time):
        ok = self._passes() and (window_start <= target_time <= window_end)
        return {"pass": ok, "reason": "mock quality check"}

    def quality_verify_proactive_multi(self, video_path, window_start, window_end, qa_history,
                                        target_question, target_answer, target_time, query_time):
        ok = self._passes() and (window_start <= target_time <= window_end + 1e-6)
        return {"pass": ok, "reason": "mock quality check"}


# --------------------------------------------------------------------------- #
# Shared base for real, frame-sampled MLLM backends
# --------------------------------------------------------------------------- #

class _FrameSampledMLLMClient(MLLMClient):
    """
    Base class for real MLLM backends that consume a *frame-sampled* view of
    the prepared (2 FPS / H.264) video.

    Every Stage 2/3/5 method below is implemented once here, in terms of a
    single backend-specific primitive:

        _call(text_prompt, video_path=None, start_s=0.0, end_s=None) -> str

    A concrete subclass only sets up its SDK and implements `_call` (which
    chat API to hit, and how to package the sampled frames for that
    provider). Frames are sampled with OpenCV since chat APIs take images,
    not raw video containers; for long ranges we send at most
    `max_frames_per_call` frames spread uniformly over the requested time
    range, which keeps token usage bounded.

    Requires `opencv-python-headless` for frame sampling.
    """

    def __init__(self, max_frames_per_call: int = 16, max_tokens: int = 2048,
                 frame_max_side: Optional[int] = None):
        self.max_frames_per_call = max_frames_per_call
        self.max_tokens = max_tokens
        # If set, downscale each sampled frame so its longest side <= this many
        # pixels before JPEG-encoding. Fewer pixels => far fewer image tokens =>
        # much faster inference (critical for CPU-served open VLMs). None sends
        # native resolution.
        self.frame_max_side = frame_max_side

    # -- frame sampling (backend-independent) -------------------------------
    def _sample_frames_b64(self, video_path: str, start_s: float = 0.0,
                            end_s: Optional[float] = None) -> List[str]:
        """Return up to `max_frames_per_call` base64-JPEG frames uniformly
        sampled from [start_s, end_s] of the video (no data: URI prefix)."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0.0
        end_s = duration if end_s is None else min(end_s, duration)
        start_s = max(0.0, start_s)

        n = self.max_frames_per_call
        if end_s <= start_s:
            timestamps = [start_s]
        else:
            timestamps = [start_s + i * (end_s - start_s) / max(1, n - 1) for i in range(n)]

        b64_frames = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            if self.frame_max_side:
                h, w = frame.shape[:2]
                longest = max(h, w)
                if longest > self.frame_max_side:
                    scale = self.frame_max_side / float(longest)
                    frame = cv2.resize(
                        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                        interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                b64_frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cap.release()
        return b64_frames

    # -- backend primitive --------------------------------------------------
    @abstractmethod
    def _call(self, text_prompt: str, video_path: Optional[str] = None,
              start_s: float = 0.0, end_s: Optional[float] = None) -> str:
        """Send `text_prompt` (plus frames sampled from the video range, if
        `video_path` is given) to the backing model; return raw text."""
        ...

    # ---- Stage 2 ----
    def segment_scenes(self, video_path, fps, duration_s):
        prompt = prompts.SCENE_SEGMENTATION_PROMPT.format(fps=fps, duration=duration_s)
        return _extract_json(self._call(prompt, video_path, 0.0, duration_s))

    def generate_candidate_qas(self, video_path, scenes, max_qas):
        prompt = prompts.CANDIDATE_QA_RT_PROACTIVE_PROMPT.format(
            scenes_json=json.dumps(scenes, ensure_ascii=False), max_qas=max_qas)
        return _extract_json(self._call(prompt, video_path))

    def verify_rt_qa(self, video_path, question, answer, a_time):
        prompt = prompts.VERIFY_RT_QA_PROMPT.format(a_time=a_time, question=question, answer=answer)
        return _extract_json(self._call(prompt, video_path, 0.0, a_time))

    def verify_proactive_qa(self, video_path, question, answer, q_time, a_time):
        prompt = prompts.VERIFY_PROACTIVE_QA_PROMPT.format(
            a_time=a_time, question=question, answer=answer, q_time=q_time)
        return _extract_json(self._call(prompt, video_path, 0.0, a_time))

    def generate_multi_candidate_questions(self, video_path, clip_start, clip_end, scene_description, max_questions):
        prompt = prompts.CANDIDATE_MULTI_QUESTION_PROMPT.format(
            clip_start=clip_start, clip_end=clip_end,
            scene_description=scene_description, max_questions=max_questions)
        return _extract_json(self._call(prompt, video_path, clip_start, clip_end))

    def check_multi_answerable(self, video_path, clip_start, clip_end, question, q_time):
        prompt = prompts.CHECK_MULTI_ANSWERABLE_PROMPT.format(
            clip_start=clip_start, clip_end=clip_end, question=question, q_time=q_time)
        return _extract_json(self._call(prompt, video_path, clip_start, clip_end))

    def generate_multi_answers(self, video_path, clip_start, clip_end, question, q_time):
        prompt = prompts.GENERATE_MULTI_ANSWERS_PROMPT.format(
            clip_start=clip_start, clip_end=clip_end, question=question, q_time=q_time)
        return _extract_json(self._call(prompt, video_path, q_time, clip_end))

    def verify_multi_answer(self, video_path, question, answer, timestamp):
        prompt = prompts.VERIFY_MULTI_ANSWER_PROMPT.format(
            timestamp=timestamp, question=question, answer=answer)
        return _extract_json(self._call(prompt, video_path, 0.0, timestamp))

    # ---- Stage 3 ----
    def generate_difficulty_siblings(self, video_path, question, t):
        prompt = prompts.GENERATE_DIFFICULTY_SIBLINGS_PROMPT.format(t=t, question=question)
        return _extract_json(self._call(prompt, video_path, 0.0, t))

    def generate_answer_for_question(self, video_path, question, t):
        prompt = prompts.GENERATE_ANSWER_FOR_QUESTION_PROMPT.format(t=t, question=question)
        return _extract_json(self._call(prompt, video_path, 0.0, t))

    def rewrite_question(self, question, template):
        prompt = prompts.REWRITE_QUESTION_PROMPT.format(template=template, question=question)
        return _extract_json(self._call(prompt))  # text-only LLM call, no video needed

    # ---- Stage 5 ----
    # The judge sees the retained video window's frames ([window_start, window_end])
    # plus the text QA history, matching Section 4.5's visual-grounding checks.
    def quality_verify_rt(self, video_path, window_start, window_end, qa_history,
                           target_question, target_answer, target_time):
        prompt = prompts.QUALITY_VERIFY_RT_PROMPT.format(
            window_start=window_start, window_end=window_end, qa_history=qa_history,
            target_time=target_time, target_answer=target_answer, target_question=target_question)
        return _extract_json(self._call(prompt, video_path, window_start, window_end))

    def quality_verify_proactive_multi(self, video_path, window_start, window_end, qa_history,
                                        target_question, target_answer, target_time, query_time):
        prompt = prompts.QUALITY_VERIFY_PROACTIVE_MULTI_PROMPT.format(
            window_start=window_start, window_end=window_end, qa_history=qa_history,
            target_time=target_time, target_answer=target_answer,
            target_question=target_question, query_time=query_time)
        return _extract_json(self._call(prompt, video_path, window_start, window_end))


# --------------------------------------------------------------------------- #
# Real implementation - Anthropic API, frame-sampled video
# --------------------------------------------------------------------------- #

class AnthropicMLLMClient(_FrameSampledMLLMClient):
    """
    Real MLLM client backed by the Anthropic Messages API.

    Frames are sampled from the (already 2 FPS, H.264) prepared video with
    OpenCV and sent as base64 image blocks alongside the text prompt, since
    the Messages API takes images rather than raw video containers.

    Requires `pip install anthropic opencv-python-headless` and an
    ANTHROPIC_API_KEY in the environment.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_frames_per_call: int = 16,
                 max_tokens: int = 2048):
        super().__init__(max_frames_per_call=max_frames_per_call, max_tokens=max_tokens)
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnthropicMLLMClient requires the `anthropic` package: pip install anthropic"
            ) from e
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def _call(self, text_prompt: str, video_path: Optional[str] = None,
              start_s: float = 0.0, end_s: Optional[float] = None) -> str:
        content: List[Dict] = []
        if video_path is not None:
            for b64 in self._sample_frames_b64(video_path, start_s, end_s):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content.append({"type": "text", "text": text_prompt})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


# --------------------------------------------------------------------------- #
# Real implementation - Aliyun Bailian (百炼) / DashScope, Qwen-VL
# --------------------------------------------------------------------------- #

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Cheaper/faster VL alternatives on Bailian, in case you want to trade quality
# for cost: "qwen3-vl-flash", "qwen-vl-plus". Strongest current default below.
DASHSCOPE_DEFAULT_MODEL = "qwen3-vl-plus"


def _dashscope_access_denied_hint(model: str, err: Exception) -> str:
    return (
        f"DashScope/Bailian denied access to model '{model}' (HTTP 403, "
        f"Model.AccessDenied).\n"
        f"The API key is valid (it can list models) but is not authorized to "
        f"*invoke* any model yet. This is an account-level setting, not a bug "
        f"in this code. To fix, in the Bailian console (bailian.console.aliyun.com):\n"
        f"  1. Complete real-name / identity verification (实名认证) — this alone "
        f"blocks all model calls until done.\n"
        f"  2. Activate the model service (开通百炼 / 开通模型), and if the specific "
        f"Qwen-VL model shows '开通' / 'Activate', click it.\n"
        f"  3. Make sure THIS key belongs to a workspace (业务空间) that has the "
        f"model authorized, and that billing / free-tier quota is enabled.\n"
        f"Once a plain text call (e.g. model 'qwen-turbo') succeeds, re-run this "
        f"pipeline.\n"
        f"Original error: {err}"
    )


class DashScopeMLLMClient(_FrameSampledMLLMClient):
    """
    Real MLLM client for Aliyun Bailian (百炼) / DashScope Qwen-VL models,
    via the OpenAI-compatible endpoint.

    Frames are sampled from the prepared 2 FPS / H.264 video with OpenCV and
    sent to the model. Qwen-VL understands a list of frames sent as a single
    ``video`` content block (better temporal grounding than independent
    images); a single frame, or a model/endpoint that rejects the ``video``
    type, falls back to ``image_url`` blocks automatically.

    Requires `pip install openai opencv-python-headless` and a DashScope /
    Bailian API key, passed as `api_key=` or via the `DASHSCOPE_API_KEY`
    (or `BAILIAN_API_KEY`) environment variable.

    Text-only calls (Stage 3 question rewriting, Stage 5 quality checks) reuse
    the same Qwen-VL model with no frames attached, matching the AnthropicMLLM
    client's behavior.
    """

    def __init__(self, model: str = DASHSCOPE_DEFAULT_MODEL, api_key: Optional[str] = None,
                 base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL, max_frames_per_call: int = 16,
                 max_tokens: int = 2048, frame_format: str = "video",
                 temperature: float = 0.2, max_retries: int = 4, request_timeout: float = 120.0,
                 frame_max_side: Optional[int] = None):
        super().__init__(max_frames_per_call=max_frames_per_call, max_tokens=max_tokens,
                         frame_max_side=frame_max_side)
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DashScopeMLLMClient requires the `openai` package (used for the "
                "OpenAI-compatible DashScope endpoint): pip install openai"
            ) from e
        from openai import OpenAI

        key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
        if not key:
            raise ValueError(
                "No DashScope/Bailian API key found. Pass api_key=... or set the "
                "DASHSCOPE_API_KEY (or BAILIAN_API_KEY) environment variable."
            )
        # We handle retries ourselves (to distinguish auth failures from transient
        # ones), so disable the SDK's own retry loop.
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=request_timeout, max_retries=0)
        self.model = model
        if frame_format not in ("video", "image_url"):
            raise ValueError("frame_format must be 'video' or 'image_url'")
        self.frame_format = frame_format
        self.temperature = temperature
        self.max_retries = max(1, max_retries)

    # -- content packaging --------------------------------------------------
    def _build_content(self, text_prompt: str, frames: List[str]) -> List[Dict]:
        content: List[Dict] = []
        if frames:
            # A single frame is not a "video"; send it as an image regardless.
            use_video = self.frame_format == "video" and len(frames) >= 2
            if use_video:
                content.append({
                    "type": "video",
                    "video": [f"data:image/jpeg;base64,{b}" for b in frames],
                })
            else:
                for b in frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b}"},
                    })
        content.append({"type": "text", "text": text_prompt})
        return content

    def _create(self, content: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""

    def _call(self, text_prompt: str, video_path: Optional[str] = None,
              start_s: float = 0.0, end_s: Optional[float] = None) -> str:
        frames = self._sample_frames_b64(video_path, start_s, end_s) if video_path is not None else []
        content = self._build_content(text_prompt, frames)

        last_err: Optional[Exception] = None
        did_image_fallback = False
        attempt = 0  # counts only genuine transient retries, not the format fallback
        while attempt < self.max_retries:
            try:
                return self._create(content)
            except Exception as e:  # noqa: BLE001 - re-raised below with context
                status = getattr(e, "status_code", None)
                # Auth / permission: never retry, surface actionable guidance.
                if status in (401, 403):
                    raise RuntimeError(_dashscope_access_denied_hint(self.model, e)) from e
                if status == 404:
                    raise RuntimeError(
                        f"DashScope model '{self.model}' not found (HTTP 404). "
                        f"Check the model id against your Bailian model list."
                    ) from e
                last_err = e
                # One-time content-format fallback (video -> per-frame image_url)
                # when the endpoint rejects the 'video' block. This does NOT
                # consume a retry attempt, so the rebuilt request is always sent
                # at least once — even when max_retries == 1, or when earlier
                # transient failures have used up the retry budget.
                if (status == 400 and frames and self.frame_format == "video"
                        and not did_image_fallback):
                    logger.warning(
                        "Endpoint rejected 'video' content (400); falling back to "
                        "per-frame image_url blocks for the rest of this run: %s", e)
                    self.frame_format = "image_url"
                    content = self._build_content(text_prompt, frames)
                    did_image_fallback = True
                    continue
                # Transient (429 / 5xx / timeout / connection): back off and retry.
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise
        assert last_err is not None
        raise last_err


# --------------------------------------------------------------------------- #
# Real implementation - any LOCAL / self-hosted OpenAI-compatible VLM server
# --------------------------------------------------------------------------- #

class OpenAICompatibleMLLMClient(DashScopeMLLMClient):
    """
    Generic OpenAI-compatible *vision* client for a local or self-hosted server:
    Ollama (`http://localhost:11434/v1`), llama.cpp `llama-server`, LM Studio,
    or vLLM. This is the backend to use for offline testing with an open-weights
    VLM (e.g. Qwen2.5-VL) when you have no cloud VL access.

    It reuses the DashScope client's OpenAI-compatible transport verbatim (same
    `/chat/completions` calls, frame sampling, retry/format-fallback and JSON
    extraction) and only changes the defaults that matter locally:
      * `base_url` points at your local server, and no real API key is required;
      * `frame_format` defaults to the standard `image_url` blocks (local servers
        implement those, not DashScope's provider-specific `video` block);
      * fewer frames per call and a long request timeout, since CPU inference of
        an open VLM is much slower than a hosted API.

    Any OpenAI-compatible endpoint works — just pass `base_url` and `model`.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "local", frame_format: str = "image_url",
                 max_frames_per_call: int = 8, max_tokens: int = 1024,
                 temperature: float = 0.2, max_retries: int = 3,
                 request_timeout: float = 900.0, frame_max_side: Optional[int] = 512):
        super().__init__(
            model=model, api_key=api_key, base_url=base_url, frame_format=frame_format,
            max_frames_per_call=max_frames_per_call, max_tokens=max_tokens,
            temperature=temperature, max_retries=max_retries, request_timeout=request_timeout,
            frame_max_side=frame_max_side,
        )
