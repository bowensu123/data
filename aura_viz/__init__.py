"""
aura_viz — a lightweight local web platform for inspecting the streaming
video-LLM training data produced by the AURA data engine.

Loads a `training_instances.jsonl` (the pipeline output), and renders each
TrainingInstance as an interactive chunk-wise streaming timeline: the mixed
Real-Time / Proactive / Multi-Response conversation, the dual-sliding-window
context, the Section 5.1 supervision mask, per-chunk loss weights, and — when
the prepared videos are on disk — the actual video frame at each timestamp.

Run it:

    python -m aura_viz --work-dir real_test/output --port 8000

then open http://localhost:8000.
"""

__all__ = ["serve"]

from .server import serve
