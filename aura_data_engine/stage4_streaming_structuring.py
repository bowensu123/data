"""
Stage 4 — Streaming Structuring (Section 4.4, with Section 4.6 "Additional
Details" and the Dual Sliding-Window Strategy of Section 3.1).

Two steps:

(A) Build the dense, chunk-wise conversational stream for a video (one
    exchange per `chunk_size_s` seconds), placing Real-Time, Proactive, and
    Multi-Response QA events at their timestamps, inserting a short
    acknowledgment immediately after each Proactive/Multi-Response query
    (Section 4.6), and mixing all three QA types from the same source video
    into one interleaved sequence ordered by timestamp.

(B) Unroll: unwind the stream into one training instance per non-silent
    assistant message, anchored at that message's timestamp, applying the
    Dual Sliding-Window Strategy (video window N, QA window M) to truncate
    context, and computing the Section 5.1 supervision mask (all silent
    turns + the final/target non-silent turn are supervised; earlier
    non-silent turns are context-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import AURADataEngineConfig
from .schema import (
    StreamChunk, TrainingInstance, VideoRecord, QAType, SILENT_TOKEN, new_id,
    RealTimeQA, ProactiveQA, MultiResponseQA,
)

ACKNOWLEDGMENTS = [
    "Got it, I'll keep watching and let you know.",
    "Sure, give me a moment to check.",
    "Okay, I'm on it.",
    "Understood, I'll follow up shortly.",
]


@dataclass
class QAGroup:
    """A user question together with all subsequent non-silent assistant
    responses to it (Section 3.1's definition of a "QA group")."""
    qa_id: str
    qa_type: QAType
    query_t: int
    response_ts: List[int] = field(default_factory=list)  # includes the ack, if any


# --------------------------------------------------------------------------- #
# (A) Building the dense mixed chunk-wise stream
# --------------------------------------------------------------------------- #

def _place(chunks: List[StreamChunk], t: int, *, user_text: Optional[str] = None,
           assistant_text: Optional[str] = None, qa_type: Optional[QAType] = None,
           qa_id: Optional[str] = None, is_ack: bool = False) -> int:
    """Write into the first available slot at-or-after `t` (linear probing to
    resolve rare timestamp collisions when mixing three QA streams), and
    return the chunk index actually used."""
    n = len(chunks)
    i = max(0, min(t, n - 1))
    while True:
        c = chunks[i]
        user_ok = user_text is None or c.user_text is None
        assistant_ok = assistant_text is None or c.is_silent
        if user_ok and assistant_ok:
            break
        i += 1
        if i >= n:
            # Fall back to the last chunk if we probe off the end (very long
            # response backlog on a short video); better to compress than drop data.
            i = n - 1
            break
    if user_text is not None:
        chunks[i].user_text = (chunks[i].user_text + " " + user_text).strip() if chunks[i].user_text else user_text
    if assistant_text is not None:
        chunks[i].assistant_text = assistant_text
        chunks[i].qa_type = qa_type
        chunks[i].qa_id = qa_id
        chunks[i].is_acknowledgment = is_ack
    return i


def build_chunk_stream(video: VideoRecord,
                        rt_qas: List[RealTimeQA],
                        proactive_qas: List[ProactiveQA],
                        multi_qas: List[MultiResponseQA],
                        cfg: AURADataEngineConfig,
                        insert_acknowledgments: bool = True):
    """Returns (chunks: List[StreamChunk], groups: Dict[qa_id, QAGroup])."""
    n_chunks = max(1, math.ceil(video.duration_s / cfg.chunk_size_s))
    chunks = [StreamChunk(t_s=t, video_id=video.video_id) for t in range(n_chunks)]
    groups: Dict[str, QAGroup] = {}

    def sec(x: float) -> int:
        return max(0, min(n_chunks - 1, int(round(x / cfg.chunk_size_s))))

    # Real-Time QA: single (question, answer) at the same chunk.
    for qa in rt_qas:
        t = sec(qa.a_time_s)
        idx = _place(chunks, t, user_text=qa.question, assistant_text=qa.answer,
                     qa_type=QAType.REAL_TIME, qa_id=qa.qa_id)
        groups[qa.qa_id] = QAGroup(qa_id=qa.qa_id, qa_type=QAType.REAL_TIME,
                                    query_t=idx, response_ts=[idx])

    # Proactive QA: question + ack at q_time, delayed answer at a_time.
    for qa in proactive_qas:
        qt = sec(qa.q_time_s)
        ack_text = qa.acknowledgment or (ACKNOWLEDGMENTS[hash(qa.qa_id) % len(ACKNOWLEDGMENTS)]
                                          if insert_acknowledgments else SILENT_TOKEN)
        q_idx = _place(chunks, qt, user_text=qa.question,
                        assistant_text=ack_text if insert_acknowledgments else None,
                        qa_type=QAType.PROACTIVE, qa_id=qa.qa_id, is_ack=True)
        at = sec(qa.a_time_s)
        at = max(at, q_idx + 1) if at <= q_idx else at
        a_idx = _place(chunks, at, assistant_text=qa.answer,
                        qa_type=QAType.PROACTIVE, qa_id=qa.qa_id, is_ack=False)
        response_ts = [a_idx] if not insert_acknowledgments else [q_idx, a_idx]
        groups[qa.qa_id] = QAGroup(qa_id=qa.qa_id, qa_type=QAType.PROACTIVE,
                                    query_t=q_idx, response_ts=response_ts)

    # Multi-Response QA: question + ack at q_time, then each answer at its own timestamp.
    for qa in multi_qas:
        qt = sec(qa.q_time_s)
        ack_text = qa.acknowledgment or (ACKNOWLEDGMENTS[hash(qa.qa_id) % len(ACKNOWLEDGMENTS)]
                                          if insert_acknowledgments else SILENT_TOKEN)
        q_idx = _place(chunks, qt, user_text=qa.question,
                        assistant_text=ack_text if insert_acknowledgments else None,
                        qa_type=QAType.MULTI_RESPONSE, qa_id=qa.qa_id, is_ack=True)
        response_ts = [q_idx] if insert_acknowledgments else []
        last = q_idx
        for ans in qa.answers:
            t = sec(ans.timestamp_s)
            t = max(t, last + 1) if t <= last else t
            idx = _place(chunks, t, assistant_text=ans.text,
                         qa_type=QAType.MULTI_RESPONSE, qa_id=qa.qa_id, is_ack=False)
            response_ts.append(idx)
            last = idx
        groups[qa.qa_id] = QAGroup(qa_id=qa.qa_id, qa_type=QAType.MULTI_RESPONSE,
                                    query_t=q_idx, response_ts=response_ts)

    return chunks, groups


# --------------------------------------------------------------------------- #
# (B) Dual sliding-window unrolling
# --------------------------------------------------------------------------- #

def _text_repr(c: StreamChunk) -> str:
    parts = []
    if c.user_text:
        parts.append(f"User: {c.user_text}")
    if not c.is_silent:
        parts.append(f"Assistant: {c.assistant_text}")
    return " | ".join(parts)


def unroll_training_instances(video: VideoRecord, chunks: List[StreamChunk],
                               groups: Dict[str, QAGroup], cfg: AURADataEngineConfig
                               ) -> List[TrainingInstance]:
    instances: List[TrainingInstance] = []
    n = len(chunks)
    N = cfg.video_window_n_s // cfg.chunk_size_s
    M = cfg.qa_window_m_groups

    # Every chunk index that is a non-silent assistant turn is its own unroll target.
    target_indices = [i for i, c in enumerate(chunks) if not c.is_silent]

    for target_idx in target_indices:
        window_start = max(0, target_idx - N + 1)
        video_window = chunks[window_start: target_idx + 1]  # full multimodal chunks

        # Which QA groups are entirely "outside" the video window (query before window_start)?
        outside_groups = [g for g in groups.values() if g.query_t < window_start]
        outside_groups.sort(key=lambda g: g.query_t)
        kept_outside = outside_groups[-M:] if M > 0 else []

        history_items: List[StreamChunk] = []
        for g in kept_outside:
            q_chunk = chunks[g.query_t]
            history_items.append(StreamChunk(
                t_s=q_chunk.t_s, video_id=video.video_id, user_text=q_chunk.user_text,
                assistant_text=SILENT_TOKEN, qa_type=g.qa_type, qa_id=g.qa_id, text_only=True,
            ))
            # Only response timestamps still outside the window are kept as text;
            # ones that now fall inside the video window are already represented there.
            for r in g.response_ts:
                if r < window_start:
                    rc = chunks[r]
                    history_items.append(StreamChunk(
                        t_s=rc.t_s, video_id=video.video_id, user_text=None,
                        assistant_text=rc.assistant_text, qa_type=g.qa_type, qa_id=g.qa_id,
                        is_acknowledgment=rc.is_acknowledgment, text_only=True,
                    ))
        history_items.sort(key=lambda c: c.t_s)

        full_sequence = history_items + list(video_window)
        target_pos = len(full_sequence) - 1  # target is always the last chunk (== chunks[target_idx])

        n_silent = sum(1 for c in video_window if c.is_silent)  # text-only history has no silent turns
        mask = []
        for pos, c in enumerate(full_sequence):
            if pos == target_pos:
                mask.append(True)
            elif (not c.text_only) and c.is_silent:
                mask.append(True)
            else:
                mask.append(False)

        target_chunk = chunks[target_idx]
        instances.append(TrainingInstance(
            instance_id=new_id("inst"),
            video_id=video.video_id,
            source_qa_type=target_chunk.qa_type,
            source_qa_id=target_chunk.qa_id,
            target_chunk_index=target_pos,
            chunks=full_sequence,
            supervision_mask=mask,
            n_silent_supervised=max(1, n_silent),  # avoid div-by-zero in the loss weight 1/N_silent
        ))

    return instances


def run_streaming_structuring(video: VideoRecord, rt_qas: List[RealTimeQA],
                               proactive_qas: List[ProactiveQA],
                               multi_qas: List[MultiResponseQA],
                               cfg: AURADataEngineConfig) -> List[TrainingInstance]:
    chunks, groups = build_chunk_stream(video, rt_qas, proactive_qas, multi_qas, cfg)
    return unroll_training_instances(video, chunks, groups, cfg)
