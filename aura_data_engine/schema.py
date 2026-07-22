"""
Core data structures for the AURA Coarse-to-Fine Streaming Data Engine.

These mirror the objects described in Section 4 of the AURA paper
(arXiv:2604.04184): scene segments, the three QA taxonomies (Real-Time,
Proactive, Multi-Response), the chunk-wise conversational stream, and the
final unrolled training instances produced by Streaming Structuring
(Section 4.4) together with the supervision mask used by the
Silent-Speech Balanced Loss (Section 5.1).

Everything here is plain-old-data (dataclasses) + (de)serialization helpers
so the pipeline stages can be tested independently and the final output can
be dumped as JSONL for training.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


SILENT_TOKEN = "<|silent|>"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Video preparation (Section 4.1)
# --------------------------------------------------------------------------- #

@dataclass
class VideoRecord:
    video_id: str
    src_path: str                 # original path as collected
    prepared_path: str = ""       # path after resample/re-encode to 2 FPS H.264
    domain: str = "unknown"       # sports / vlogs / documentaries / ... (Fig. 5 right)
    fps: float = 2.0
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    corrupted: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Scene segmentation (shared by both QA-synthesis sub-pipelines, Section 4.2)
# --------------------------------------------------------------------------- #

@dataclass
class Scene:
    scene_id: str
    video_id: str
    start_s: float
    end_s: float
    description: str


# --------------------------------------------------------------------------- #
# Streaming QA taxonomy (Section 3.2)
# --------------------------------------------------------------------------- #

class QAType(str, Enum):
    REAL_TIME = "real_time"
    PROACTIVE = "proactive"
    MULTI_RESPONSE = "multi_response"


class Difficulty(str, Enum):
    """Difficulty ladder used by RT-QA refinement (Section 4.3)."""
    L1_PERCEPTION = "perception"       # simple perceptual recognition
    L2_RECOGNITION = "recognition"
    L3_UNDERSTANDING = "understanding"
    L4_REASONING = "reasoning"
    L5_ADVANCED_REASONING = "advanced_reasoning"


@dataclass
class RealTimeQA:
    """Section 3.2 (1): single immediate response, q_time == a_time."""
    qa_id: str
    video_id: str
    question: str
    answer: str
    q_time_s: float
    a_time_s: float
    difficulty: Difficulty = Difficulty.L2_RECOGNITION
    verified: bool = False
    # the 4 sibling questions generated during refinement at the same timestamp,
    # kept for traceability even though only one is sampled into the final set
    difficulty_siblings: List["RealTimeQA"] = field(default_factory=list)

    def __post_init__(self):
        assert abs(self.q_time_s - self.a_time_s) < 1e-6, (
            "Real-Time QA must have q_time == a_time"
        )


@dataclass
class ProactiveQA:
    """Section 3.2 (2): silent after query, single delayed response."""
    qa_id: str
    video_id: str
    question: str
    answer: str
    q_time_s: float
    a_time_s: float
    acknowledgment: str = ""   # short ack inserted right after the query (Section 4.6)
    verified: bool = False

    def __post_init__(self):
        # Section 4.2: "for Proactive QA, the question timestamp precedes the
        # answer timestamp" — i.e. strictly a_time > q_time (the model stays
        # silent after the query and answers only later).
        assert self.a_time_s > self.q_time_s, (
            "Proactive QA requires the answer timestamp to strictly follow the question timestamp"
        )


@dataclass
class MultiResponseAnswer:
    text: str
    timestamp_s: float
    verified: bool = False


@dataclass
class MultiResponseQA:
    """Section 3.2 (3): one question, several valid answers over time."""
    qa_id: str
    video_id: str
    question: str
    q_time_s: float
    clip_start_s: float
    clip_end_s: float
    answers: List[MultiResponseAnswer] = field(default_factory=list)
    acknowledgment: str = ""


# --------------------------------------------------------------------------- #
# Chunk-wise conversational format (Section 3.1, "Chunk-wise Conversational Format")
# --------------------------------------------------------------------------- #

@dataclass
class StreamChunk:
    """
    One second (by default) of video packaged as a single user/assistant
    exchange, per the chunk-wise conversational format:
      - user message   = video chunk [+ question text if a query lands here]
      - assistant msg   = response text, or SILENT_TOKEN
    """
    t_s: int                       # chunk index in seconds since stream start
    video_id: str
    user_text: Optional[str] = None            # question text, if any, else None
    assistant_text: str = SILENT_TOKEN          # response text, else SILENT_TOKEN
    qa_type: Optional[QAType] = None            # which taxonomy this exchange belongs to
    qa_id: Optional[str] = None                 # back-reference to the source QA item
    is_acknowledgment: bool = False             # True if assistant_text is just an ack
    text_only: bool = False                     # True for QA-window history entries whose
                                                 # video chunk + silent tokens were discarded
                                                 # per the dual sliding-window strategy

    @property
    def is_silent(self) -> bool:
        return self.assistant_text == SILENT_TOKEN

    @property
    def has_query(self) -> bool:
        return self.user_text is not None


# --------------------------------------------------------------------------- #
# Unrolled training instance produced by Streaming Structuring (Section 4.4)
# and scored by Quality Verification (Section 4.5)
# --------------------------------------------------------------------------- #

@dataclass
class TrainingInstance:
    """
    One unrolled, window-truncated training sample.

    `chunks` is the ordered list of StreamChunk objects that survive the
    dual sliding-window truncation (video window N, QA window M), anchored
    at `target_chunk_index` — i.e. the position of the target (supervised)
    non-silent assistant message inside `chunks`.

    `supervision_mask[i]` corresponds 1:1 with `chunks[i]` and is True iff
    that assistant turn should receive loss (silent turns + the final
    non-silent turn), matching m_t in Eq. (1) of the paper.
    """
    instance_id: str
    video_id: str
    source_qa_type: QAType
    source_qa_id: str
    target_chunk_index: int
    chunks: List[StreamChunk]
    supervision_mask: List[bool]
    n_silent_supervised: int = 0     # N_silent used for the loss re-weighting
    quality_passed: Optional[bool] = None
    quality_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_qa_type"] = self.source_qa_type.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainingInstance":
        d = dict(d)
        d["source_qa_type"] = QAType(d["source_qa_type"])
        d["chunks"] = [StreamChunk(**{**c, "qa_type": QAType(c["qa_type"]) if c.get("qa_type") else None})
                        for c in d["chunks"]]
        return TrainingInstance(**d)


def dump_jsonl(items: List[Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if hasattr(it, "to_dict"):
                obj = it.to_dict()
            elif hasattr(it, "__dict__") or hasattr(it, "__dataclass_fields__"):
                obj = asdict(it)
            else:
                obj = it
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
