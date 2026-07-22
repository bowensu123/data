"""
Stage 3 — QA Refinement (Section 4.3).

"Real-Time QA tends to exhibit limited diversity in difficulty levels, while
Proactive QA and Multi-Response QA often show limited diversity in question
phrasing."

RT-QA refinement:
  For each original QA at timestamp t, generate 4 additional questions at
  the same t spanning increasing difficulty (perception -> advanced
  reasoning), giving 5 candidates total (original + 4). Sample ONE using a
  balanced sampling ratio (implemented as uniform-over-5, i.e. each
  difficulty level, including the original, is equally likely to be
  selected in expectation across the dataset). Generate a fresh answer for
  the sampled question against the same video prefix.

Proactive / Multi-Response refinement:
  Rewrite the question into a different, semantically-equivalent phrasing
  by sampling one of a predefined set of style templates and asking an LLM
  to rewrite while holding entities/actions/temporal references fixed.

Refinement is best-effort: if the model returns unusable output for a
sibling / rewrite / re-answer, that step degrades gracefully to the original
QA rather than dropping or crashing it (see parse_utils).
"""

from __future__ import annotations

import logging
import random
from typing import List

from .config import AURADataEngineConfig
from .llm_client import MLLMClient
from .parse_utils import as_float, as_text, as_difficulty
from .prompts import PROACTIVE_QUESTION_TEMPLATES, MULTI_RESPONSE_QUESTION_TEMPLATES
from .schema import RealTimeQA, ProactiveQA, MultiResponseQA, VideoRecord

logger = logging.getLogger("aura_data_engine.stage3")


def refine_real_time_qa(client: MLLMClient, video: VideoRecord, qa: RealTimeQA,
                         cfg: AURADataEngineConfig, rng: random.Random) -> RealTimeQA:
    """Produce the final RT-QA: original + 4 difficulty siblings, sample 1, re-answer."""
    if cfg.difficulty_sampling != "uniform":
        raise NotImplementedError(f"Unsupported difficulty_sampling: {cfg.difficulty_sampling}")

    try:
        siblings_raw = client.generate_difficulty_siblings(
            video.prepared_path, qa.question, qa.a_time_s)
    except Exception:  # noqa: BLE001 - refinement must not abort the QA
        logger.debug("difficulty-sibling generation failed; using original only", exc_info=True)
        siblings_raw = []

    siblings: List[RealTimeQA] = []
    for i, s in enumerate(siblings_raw if isinstance(siblings_raw, list) else []):
        if not isinstance(s, dict):
            continue
        q = as_text(s.get("question"))
        if not q:
            continue
        siblings.append(RealTimeQA(
            qa_id=f"{qa.qa_id}_sib{i}", video_id=video.video_id,
            question=q, answer="",  # answer filled only if this one is chosen
            q_time_s=qa.a_time_s, a_time_s=qa.a_time_s,
            difficulty=as_difficulty(s.get("difficulty")),
        ))

    candidates = [qa] + siblings  # up to 5 candidates total, per the paper
    chosen = rng.choice(candidates)

    # Section 4.3: "After the question is selected, we prompt the MLLM to generate
    # an answer aligned with the sampled question based on the same video prefix" —
    # done for whichever of the five was sampled, including the original.
    question = chosen.question
    difficulty = chosen.difficulty
    answer = ""
    try:
        ans = client.generate_answer_for_question(video.prepared_path, question, qa.a_time_s)
        answer = as_text(ans.get("answer") if isinstance(ans, dict) else ans)
    except Exception:  # noqa: BLE001
        logger.debug("re-answer generation failed for sampled RT question", exc_info=True)
    if not answer:
        # Fallback only on model failure: keep the original (already-verified) QA
        # rather than emit an empty answer.
        question, answer, difficulty = qa.question, qa.answer, qa.difficulty

    final = RealTimeQA(
        qa_id=qa.qa_id, video_id=video.video_id,
        question=question, answer=answer,
        q_time_s=qa.q_time_s, a_time_s=qa.a_time_s,
        difficulty=difficulty, verified=qa.verified,
    )
    final.difficulty_siblings = siblings
    return final


def _rewrite_or_keep(client: MLLMClient, question: str, templates, rng: random.Random) -> str:
    """Rewrite `question` via a sampled style template, keeping the original if
    the model returns nothing usable."""
    template = rng.choice(templates)
    try:
        rewritten = client.rewrite_question(question, template)
        new_q = as_text(rewritten.get("question") if isinstance(rewritten, dict) else rewritten)
    except Exception:  # noqa: BLE001
        logger.debug("question rewrite failed; keeping original", exc_info=True)
        new_q = ""
    return new_q or question


def refine_proactive_qa(client: MLLMClient, qa: ProactiveQA, rng: random.Random) -> ProactiveQA:
    return ProactiveQA(
        qa_id=qa.qa_id, video_id=qa.video_id,
        question=_rewrite_or_keep(client, qa.question, PROACTIVE_QUESTION_TEMPLATES, rng),
        answer=qa.answer, q_time_s=qa.q_time_s, a_time_s=qa.a_time_s,
        acknowledgment=qa.acknowledgment, verified=qa.verified,
    )


def refine_multi_response_qa(client: MLLMClient, qa: MultiResponseQA, rng: random.Random) -> MultiResponseQA:
    return MultiResponseQA(
        qa_id=qa.qa_id, video_id=qa.video_id,
        question=_rewrite_or_keep(client, qa.question, MULTI_RESPONSE_QUESTION_TEMPLATES, rng),
        q_time_s=qa.q_time_s, clip_start_s=qa.clip_start_s, clip_end_s=qa.clip_end_s,
        answers=qa.answers, acknowledgment=qa.acknowledgment,
    )


def run_qa_refinement(client: MLLMClient, video: VideoRecord,
                       rt_qas: List[RealTimeQA], proactive_qas: List[ProactiveQA],
                       multi_qas: List[MultiResponseQA], cfg: AURADataEngineConfig):
    rng = random.Random(cfg.random_seed)
    refined_rt = [refine_real_time_qa(client, video, qa, cfg, rng) for qa in rt_qas]
    refined_proactive = [refine_proactive_qa(client, qa, rng) for qa in proactive_qas]
    refined_multi = [refine_multi_response_qa(client, qa, rng) for qa in multi_qas]
    return refined_rt, refined_proactive, refined_multi
