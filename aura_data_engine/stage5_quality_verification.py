"""
Stage 5 — Quality Verification (Section 4.5).

"Since the previous stage reformats each sample within a truncated
streaming context window, the retained video content and contextual
history may become insufficient to support the target answer ... To
address this issue, we introduce a dedicated Quality Verification stage."

For Real-Time QA: is the target answer visually grounded, factually
correct, temporally consistent, and free of hallucination, given only the
retained video window + QA history?

For Proactive / Multi-Response QA: is the response timing appropriate, and
is the content accurate and grounded in the retained window + history?

Only samples that pass are retained; everything else is dropped before
becoming training data.
"""

from __future__ import annotations

from typing import List

from .llm_client import MLLMClient
from .schema import TrainingInstance, QAType, SILENT_TOKEN


def _qa_history_text(instance: TrainingInstance) -> str:
    lines = []
    for c in instance.chunks:
        if c.user_text:
            lines.append(f"[t={c.t_s}s] User: {c.user_text}")
        if c.assistant_text != SILENT_TOKEN:
            tag = "Ack" if c.is_acknowledgment else "Assistant"
            lines.append(f"[t={c.t_s}s] {tag}: {c.assistant_text}")
    return "\n".join(lines)


def verify_instance(client: MLLMClient, instance: TrainingInstance,
                    video_path: str = "") -> TrainingInstance:
    target = instance.chunks[instance.target_chunk_index]
    window_chunks = [c for c in instance.chunks if not c.text_only]
    window_start = window_chunks[0].t_s if window_chunks else target.t_s
    window_end = window_chunks[-1].t_s if window_chunks else target.t_s
    history_text = _qa_history_text(instance)

    # Find the originating question for this target (best-effort: nearest
    # preceding chunk sharing qa_id that carries a user_text).
    target_question = ""
    for c in reversed(instance.chunks[: instance.target_chunk_index + 1]):
        if c.qa_id == instance.source_qa_id and c.user_text:
            target_question = c.user_text
            break

    if instance.source_qa_type == QAType.REAL_TIME:
        verdict = client.quality_verify_rt(
            video_path=video_path, window_start=window_start, window_end=window_end,
            qa_history=history_text, target_question=target_question,
            target_answer=target.assistant_text, target_time=target.t_s,
        )
    else:
        query_time = window_start
        for c in instance.chunks:
            if c.qa_id == instance.source_qa_id and c.user_text:
                query_time = c.t_s
                break
        verdict = client.quality_verify_proactive_multi(
            video_path=video_path, window_start=window_start, window_end=window_end,
            qa_history=history_text, target_question=target_question,
            target_answer=target.assistant_text, target_time=target.t_s, query_time=query_time,
        )

    instance.quality_passed = bool(verdict.get("pass"))
    instance.quality_reason = verdict.get("reason", "")
    return instance


def run_quality_verification(client: MLLMClient, instances: List[TrainingInstance],
                              video_path: str = "") -> List[TrainingInstance]:
    """Verifies every instance (against the retained video window's frames) and
    returns only the ones that pass."""
    verified = [verify_instance(client, inst, video_path) for inst in instances]
    return [inst for inst in verified if inst.quality_passed]
