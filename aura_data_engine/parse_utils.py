"""
Defensive coercion for messy MLLM/LLM output.

The pipeline prompts every model to "Return ONLY a JSON ..." with specific
field types, but real models — especially small, locally-served open VLMs —
routinely deviate: a numeric timestamp comes back as a one-element list
`[5.0]`, a "MM:SS" string, a `{"seconds": 5}` object; a text field comes back
as a list of strings; a difficulty label is capitalized or paraphrased.

These helpers turn such values into the type the pipeline expects, returning a
sentinel (`None` / default) when nothing usable can be extracted so the caller
can *skip that one item* rather than crashing the whole video. Keeping this
logic in one place means every stage that consumes model output is tolerant in
the same way.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .schema import Difficulty


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float from a model-returned value.

    Handles: int/float; numeric string ("5", "5.0", "5s", "5 seconds");
    "MM:SS" / "HH:MM:SS" timecodes; a list/tuple (first coercible element);
    a dict keyed by a time-ish name. Returns `default` if nothing works.
    `bool` is intentionally rejected (True is not a timestamp).
    """
    if isinstance(x, bool) or x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        for e in x:
            v = as_float(e, None)
            if v is not None:
                return v
        return default
    if isinstance(x, dict):
        for k in ("value", "seconds", "second", "sec", "s", "time", "timestamp", "t", "ts"):
            if k in x:
                v = as_float(x[k], None)
                if v is not None:
                    return v
        return default
    if isinstance(x, str):
        s = x.strip().lower()
        # "HH:MM:SS" / "MM:SS" timecode
        if ":" in s:
            parts = s.split(":")
            try:
                total = 0.0
                for p in parts:
                    total = total * 60 + float(p)
                return total
            except ValueError:
                pass
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return default
    return default


def as_text(x: Any, default: str = "") -> str:
    """Best-effort non-empty string from a model-returned value.

    str -> itself; list/tuple -> its coercible parts joined; dict -> a common
    text-ish field; anything else -> str(x). Returns `default` if empty.
    """
    if x is None:
        return default
    if isinstance(x, str):
        return x.strip() or default
    if isinstance(x, (list, tuple)):
        parts = [as_text(e, "") for e in x]
        joined = " ".join(p for p in parts if p).strip()
        return joined or default
    if isinstance(x, dict):
        for k in ("text", "question", "answer", "value", "content", "description"):
            if k in x:
                return as_text(x[k], default)
        return default
    return str(x).strip() or default


_DIFFICULTY_KEYWORDS = [
    (Difficulty.L5_ADVANCED_REASONING, ("advanced", "complex", "multi-step", "multistep")),
    (Difficulty.L4_REASONING, ("reason", "infer", "predict", "why", "intent")),
    (Difficulty.L3_UNDERSTANDING, ("understand", "explain", "describe", "what is happening")),
    (Difficulty.L2_RECOGNITION, ("recogn", "identif", "name", "classif")),
    (Difficulty.L1_PERCEPTION, ("percept", "color", "count", "presence", "simple")),
]


def as_difficulty(x: Any, default: Difficulty = Difficulty.L2_RECOGNITION) -> Difficulty:
    """Coerce a model-returned difficulty label to a valid `Difficulty`.

    Tries the exact enum value first, then keyword matching (so "Advanced
    Reasoning", "perceptual", etc. still map sensibly), else `default`.
    """
    s = as_text(x, "").strip().lower()
    if not s:
        return default
    for d in Difficulty:
        if s == d.value:
            return d
    for d, keys in _DIFFICULTY_KEYWORDS:
        if any(k in s for k in keys):
            return d
    return default
