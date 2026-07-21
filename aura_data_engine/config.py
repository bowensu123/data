"""
Pipeline hyperparameters.

Values default to the ones reported in Section 6.1 ("Implementation
Details") of the AURA paper: chunk size 1s, video window N=30s, KV-cache
extension margin N'=15s, QA window M=10 groups, and 2 FPS video sampling
(Section 4.1). These are exposed as a dataclass so experiments can override
them without touching pipeline code.
"""

from dataclasses import dataclass


@dataclass
class AURADataEngineConfig:
    # --- Section 4.1 Video Preparation ---
    target_fps: float = 2.0
    video_codec: str = "libx264"

    # --- Section 3.1 Interactive Video Stream Context Management ---
    chunk_size_s: int = 1        # one chunk per second
    video_window_n_s: int = 30   # N: seconds of video retained in the sliding window
    prefix_margin_n_prime_s: int = 15  # N': KV-cache reuse margin (inference-time; kept
                                        # here so training-time truncation stats match)
    qa_window_m_groups: int = 10  # M: number of recent QA groups retained in history

    # --- Section 4.3 QA Refinement ---
    num_difficulty_siblings: int = 4     # +4 siblings -> 5 candidates total for RT-QA
    difficulty_sampling: str = "uniform"  # "balanced sampling ratio" -> uniform over the 5

    # --- Section 4.5 Quality Verification thresholds ---
    require_visual_grounding: bool = True
    require_temporal_consistency: bool = True

    # --- Misc ---
    random_seed: int = 0
