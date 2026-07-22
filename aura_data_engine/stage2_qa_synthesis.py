"""
Stage 2 — QA Synthesis (Section 4.2).

Two sub-pipelines, exactly as described in the paper:

Pipeline A (Real-Time + Proactive QA):
  1. MLLM scene segmentation + scene-level descriptions.
  2. MLLM proposes candidate (question, answer, q_time, a_time) pairs;
     for RT, q_time == a_time, for Proactive, q_time < a_time.
  3. MLLM verifies each candidate using only video content up to a_time
     (reasonableness / grounded answer / accurate timestamp for RT; plus
     "can the question naturally be raised at q_time" and "is there enough
     evidence by a_time" for Proactive).
  4. Only pairs that pass verification are retained.

Pipeline B (Multi-Response QA):
  1. Same scene segmentation.
  2. For each scene/clip, MLLM proposes candidate questions + timestamps.
  3. Each candidate is checked for reasonableness AND multi-answerability.
  4. For retained questions, MLLM generates multiple (answer, timestamp)
     pairs, each independently verified before being kept.

Model output is consumed defensively (see parse_utils): a candidate whose
fields cannot be coerced to the expected types is skipped, so one malformed
item from an imperfect model does not discard the whole video's work.
"""

from __future__ import annotations

import logging
from typing import List

from .llm_client import MLLMClient
from .parse_utils import as_float, as_text
from .schema import (
    Scene, RealTimeQA, ProactiveQA, MultiResponseQA, MultiResponseAnswer,
    VideoRecord, new_id,
)

logger = logging.getLogger("aura_data_engine.stage2")

MAX_CANDIDATE_QAS_PER_VIDEO = 12
MAX_CANDIDATE_QUESTIONS_PER_CLIP = 3


def _as_list(raw) -> list:
    """Model output should be a JSON list here; tolerate a single dict or a
    dict wrapping the list under a common key."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for k in ("scenes", "questions", "candidates", "answers", "items", "data", "results"):
            if isinstance(raw.get(k), list):
                return raw[k]
        return [raw]
    return []


def segment_scenes(client: MLLMClient, video: VideoRecord) -> List[Scene]:
    raw = client.segment_scenes(video.prepared_path, video.fps, video.duration_s)
    scenes = []
    for r in _as_list(raw):
        if not isinstance(r, dict):
            continue
        start = as_float(r.get("start_s"))
        end = as_float(r.get("end_s"))
        desc = as_text(r.get("description"))
        if start is None or end is None or end <= start:
            logger.debug("skipping malformed scene: %r", r)
            continue
        scenes.append(Scene(
            scene_id=new_id("scene"), video_id=video.video_id,
            start_s=start, end_s=end, description=desc,
        ))
    return scenes


# --------------------------------------------------------------------------- #
# Pipeline A: Real-Time + Proactive QA
# --------------------------------------------------------------------------- #

def synthesize_rt_and_proactive_qas(client: MLLMClient, video: VideoRecord,
                                     scenes: List[Scene],
                                     max_qas: int = MAX_CANDIDATE_QAS_PER_VIDEO):
    """Returns (List[RealTimeQA], List[ProactiveQA]), verification-filtered."""
    scenes_json = [{"start_s": s.start_s, "end_s": s.end_s, "description": s.description}
                   for s in scenes]
    candidates = client.generate_candidate_qas(video.prepared_path, scenes_json, max_qas)

    rt_qas: List[RealTimeQA] = []
    proactive_qas: List[ProactiveQA] = []

    for c in _as_list(candidates):
        if not isinstance(c, dict):
            continue
        try:
            c_type = as_text(c.get("type")).lower()
            question = as_text(c.get("question"))
            answer = as_text(c.get("answer"))
            q_time = as_float(c.get("q_time_s"))
            a_time = as_float(c.get("a_time_s"))
            if not question or not answer or q_time is None or a_time is None:
                logger.debug("skipping RT/proactive candidate with missing fields: %r", c)
                continue

            if "real_time" in c_type or c_type in ("rt", "realtime", "real-time"):
                # RT is answered immediately at the question time by definition.
                # Small models often fill a_time with the scene end instead of
                # q_time; enforce the RT invariant by construction rather than
                # dropping an otherwise-valid candidate.
                a_time = q_time
                verdict = client.verify_rt_qa(video.prepared_path, question, answer, a_time)
                if isinstance(verdict, dict) and verdict.get("pass"):
                    rt_qas.append(RealTimeQA(
                        qa_id=new_id("rtqa"), video_id=video.video_id,
                        question=question, answer=answer, q_time_s=q_time, a_time_s=a_time,
                        verified=True,
                    ))
            elif "proactive" in c_type:
                if a_time <= q_time:
                    continue  # Section 4.2: Proactive requires a_time strictly > q_time
                verdict = client.verify_proactive_qa(
                    video.prepared_path, question, answer, q_time, a_time)
                if isinstance(verdict, dict) and verdict.get("pass"):
                    proactive_qas.append(ProactiveQA(
                        qa_id=new_id("proqa"), video_id=video.video_id,
                        question=question, answer=answer, q_time_s=q_time, a_time_s=a_time,
                        verified=True,
                    ))
            # unknown type -> silently dropped
        except Exception:  # noqa: BLE001 - never let one bad candidate abort the video
            logger.debug("skipping RT/proactive candidate that raised: %r", c, exc_info=True)
            continue

    return rt_qas, proactive_qas


# --------------------------------------------------------------------------- #
# Pipeline B: Multi-Response QA
# --------------------------------------------------------------------------- #

def synthesize_multi_response_qas(client: MLLMClient, video: VideoRecord,
                                   scenes: List[Scene],
                                   max_questions_per_clip: int = MAX_CANDIDATE_QUESTIONS_PER_CLIP
                                   ) -> List[MultiResponseQA]:
    results: List[MultiResponseQA] = []

    for scene in scenes:
        candidates = client.generate_multi_candidate_questions(
            video.prepared_path, scene.start_s, scene.end_s, scene.description,
            max_questions_per_clip)

        for cand in _as_list(candidates):
            if not isinstance(cand, dict):
                continue
            try:
                question = as_text(cand.get("question"))
                q_time = as_float(cand.get("q_time_s"))
                if not question or q_time is None:
                    logger.debug("skipping multi candidate with missing fields: %r", cand)
                    continue
                if not (scene.start_s <= q_time <= scene.end_s):
                    continue

                check = client.check_multi_answerable(
                    video.prepared_path, scene.start_s, scene.end_s, question, q_time)
                if not (isinstance(check, dict) and check.get("pass")):
                    continue

                raw_answers = client.generate_multi_answers(
                    video.prepared_path, scene.start_s, scene.end_s, question, q_time)

                verified_answers: List[MultiResponseAnswer] = []
                seen_ts = set()
                for ra in _as_list(raw_answers):
                    if not isinstance(ra, dict):
                        continue
                    ts = as_float(ra.get("timestamp_s"))
                    text = as_text(ra.get("text"))
                    if ts is None or not text or not (q_time <= ts <= scene.end_s):
                        continue
                    # Section 4.2: the multiple answers must be at *different*
                    # timestamps — drop duplicates (same chunk-second).
                    ts_key = round(ts)
                    if ts_key in seen_ts:
                        continue
                    v = client.verify_multi_answer(video.prepared_path, question, text, ts)
                    if isinstance(v, dict) and v.get("pass"):
                        seen_ts.add(ts_key)
                        verified_answers.append(
                            MultiResponseAnswer(text=text, timestamp_s=ts, verified=True))

                # A Multi-Response QA needs at least 2 distinct verified timestamps to be
                # meaningfully "multi" — otherwise it degenerates to a Real-Time QA.
                if len(verified_answers) >= 2:
                    results.append(MultiResponseQA(
                        qa_id=new_id("mrqa"), video_id=video.video_id,
                        question=question, q_time_s=q_time,
                        clip_start_s=scene.start_s, clip_end_s=scene.end_s,
                        answers=sorted(verified_answers, key=lambda a: a.timestamp_s),
                    ))
            except Exception:  # noqa: BLE001 - one bad candidate must not abort the video
                logger.debug("skipping multi candidate that raised: %r", cand, exc_info=True)
                continue

    return results


def run_qa_synthesis(client: MLLMClient, video: VideoRecord):
    """Full Stage 2 entry point for one video.

    Returns dict with keys "scenes", "real_time", "proactive", "multi_response".
    """
    scenes = segment_scenes(client, video)
    rt_qas, proactive_qas = synthesize_rt_and_proactive_qas(client, video, scenes)
    multi_qas = synthesize_multi_response_qas(client, video, scenes)
    return {
        "scenes": scenes,
        "real_time": rt_qas,
        "proactive": proactive_qas,
        "multi_response": multi_qas,
    }
