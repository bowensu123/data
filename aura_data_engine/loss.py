"""
Silent-Speech Balanced Loss (Section 5.1), Eq. (1).

    L = - (1 / sum_t m_t) * sum_t  m_t * w_t * log p_theta(y_t | x, y_<t)

    w_t = 1 / N_silent   if y_t belongs to a silent assistant message
    w_t = 1              otherwise (the single supervised non-silent message)

This is a *training-time* concern, not data generation per se, but it is
determined entirely by the mask/weights that Stage 4 already computes for
each TrainingInstance, so it's included here as a thin reference
implementation showing how to turn `TrainingInstance.supervision_mask` /
`n_silent_supervised` into per-token loss weights for an arbitrary
tokenized target sequence.

This module is framework-agnostic (pure Python / lists) so it can be
plugged into PyTorch, JAX, etc. without imposing a dependency here.
"""

from __future__ import annotations

from typing import List

from .schema import TrainingInstance, SILENT_TOKEN


def per_chunk_weights(instance: TrainingInstance) -> List[float]:
    """
    Returns one weight per chunk in `instance.chunks`, aligned with
    `instance.supervision_mask`:
      - 0.0 for chunks not supervised (m_t = 0)
      - 1.0 for the single supervised non-silent (target) chunk
      - 1 / N_silent for each supervised silent chunk
    A downstream tokenizer/collator should broadcast each chunk's weight to
    all of that chunk's assistant-side target tokens.
    """
    n_silent = max(1, instance.n_silent_supervised)
    weights = []
    for chunk, supervised in zip(instance.chunks, instance.supervision_mask):
        if not supervised:
            weights.append(0.0)
        elif chunk.assistant_text == SILENT_TOKEN:
            weights.append(1.0 / n_silent)
        else:
            weights.append(1.0)
    return weights


def summarize_supervision(instances: List[TrainingInstance]) -> dict:
    """Quick sanity-check aggregate stats: how much of the supervision signal
    is silent vs. speech, dataset-wide."""
    n_silent_tokens = 0
    n_speech_tokens = 0
    for inst in instances:
        for chunk, supervised in zip(inst.chunks, inst.supervision_mask):
            if not supervised:
                continue
            if chunk.assistant_text == SILENT_TOKEN:
                n_silent_tokens += 1
            else:
                n_speech_tokens += 1
    total = max(1, n_silent_tokens + n_speech_tokens)
    return {
        "n_supervised_silent_turns": n_silent_tokens,
        "n_supervised_speech_turns": n_speech_tokens,
        "silent_fraction": n_silent_tokens / total,
    }
