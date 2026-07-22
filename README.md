# AURA Coarse-to-Fine Streaming Data Engine — reproduction

A from-scratch, runnable reproduction of the **data generation pipeline** (Section 4,
Figure 3) of

> Lu, Bo, Chen, Li, Guo, Guan, Liu, Xu, Sun, Sun, Liu, Li. *AURA: Always-On Understanding
> and Real-Time Assistance via Video Streams.* arXiv:2604.04184, 2026.

**Why this exists as a from-scratch implementation:** the paper's [official
repo](https://github.com/aurateam2026/AURA) ships the *inference/deployment* stack
(vLLM serving, ASR, TTS, context-management for live streaming) and a benchmark-eval
harness, but the five-stage data engine itself was never released — there is no
`data_engine/` anywhere in that repo. Everything under `aura_data_engine/` here is
reconstructed directly from the algorithmic description in Sections 3.1, 3.2, 4, and
5.1 of the paper (fetched from arXiv on 2026-07-20).

> **How close is this to the paper?** See **[COMPARISON.md](COMPARISON.md)** for an
> honest account of what is faithfully reproduced (algorithm, hyperparameters, loss)
> vs. what is not (the dataset, model training, and the strong generation MLLM — which
> the paper never names), plus what this project adds on top.

## What's faithfully reproduced vs. what's a documented judgment call

The paper specifies *what* each stage must accomplish, not implementation-level
details like exact prompts or collision handling when timestamps from three QA
streams coincide. Everything below is implemented to satisfy the stated requirement
exactly; where the paper is silent I made an explicit, documented choice rather than
guessing silently:

| Paper requirement | Where | Design choice made here |
|---|---|---|
| Resample to 2 FPS, re-encode H.264, drop undecodable videos | §4.1 | `ffmpeg`/`ffprobe` wrapper; corrupted sources are filtered before Stage 2 (`stage1_video_preparation.py`) |
| RT-QA: `q_time == a_time`; Proactive: `q_time < a_time` (well, `<=`, see below) | §4.2 | Enforced as dataclass invariants in `schema.py` |
| Two-stage verify-then-filter for RT/Proactive | §4.2 | `stage2_qa_synthesis.py::synthesize_rt_and_proactive_qas` |
| Multi-Response: reasonable + multi-answerable check, then per-answer verification | §4.2 | `synthesize_multi_response_qas`; I additionally require **≥2 verified timestamps** to keep an item as Multi-Response (otherwise it degenerates to a single-answer RT-QA — the paper doesn't state a minimum, this is the natural one) |
| RT-QA: +4 difficulty siblings, "balanced sampling ratio" of 1-of-5, re-answer if not original | §4.3 | Implemented as **uniform** sampling over the 5 candidates (`stage3_qa_refinement.py`); "balanced" most plausibly means each difficulty level is equally likely across the corpus, which uniform-per-item achieves in expectation |
| Proactive/Multi: template-based question rewriting preserving entities/actions/time refs | §4.3 | Small template pools in `prompts.py` (`PROACTIVE_QUESTION_TEMPLATES`, `MULTI_RESPONSE_QUESTION_TEMPLATES`) — the paper doesn't publish its template set, so these are illustrative placeholders you should replace with your own |
| Chunk-wise conversational format, 1 chunk/second | §3.1, §4.4 | `stage4_streaming_structuring.py::build_chunk_stream` |
| Dual sliding window: video window N, QA window M, "full history if duration ≤ N" | §3.1 | `unroll_training_instances` — the `max(0, target_idx - N + 1)` clamp naturally reduces to "keep everything" when duration ≤ N, no special-casing needed |
| Unroll into one sample per non-silent assistant turn, anchored at its timestamp | §4.4 | One `TrainingInstance` per non-silent chunk, including acknowledgment turns (the paper doesn't exclude them, and Section 4.6 treats acks as real assistant utterances) |
| Supervise all silent turns + only the final non-silent turn; `w_silent = 1/N_silent` | §5.1, Eq. (1) | `supervision_mask` computed in `unroll_training_instances`; weights derived from it in `loss.py::per_chunk_weights` |
| Insert a short acknowledgment right after each Proactive/Multi query | §4.6 | Implemented as the assistant turn *at the query's own chunk* (rather than a separate later chunk) — the most natural reading of "immediately after," since each chunk pairs exactly one user/assistant exchange |
| Mix RT + Proactive + Multi from the same video by timestamp | §4.6 | `build_chunk_stream` places all three onto one dense per-second array; rare timestamp collisions are resolved by linear-probing to the next free slot (`_place()`) — not specified by the paper, since prompt-level collision handling isn't discussed |
| For an "outside" QA group, drop video chunks + silent tokens, keep only text | §3.1 | `unroll_training_instances`: a group counts as "outside" if its **query** precedes the video window; only response turns still outside the window are kept as text (ones that now fall inside the window are already present there, avoiding duplication) |

Every algorithmic choice the paper *specifies* is aligned to the paper (see
[COMPARISON.md](COMPARISON.md)): Proactive QA enforces `a_time > q_time`
(§4.2, "the question timestamp precedes the answer timestamp"); RT refinement
re-generates the answer for whichever of the 5 candidate questions is sampled
(§4.3); Multi-Response answers must be at distinct timestamps (§4.2); and
**Stage 5 quality verification judges against the retained video window's actual
frames** plus the QA history (§4.5), not text alone. What remains different is
only what the paper does not publish (exact prompts / rewrite templates /
verification rubric) or what needs a strong MLLM + large corpus + training.

## Layout

```
aura_data_engine/
  schema.py                      # VideoRecord, Scene, RealTimeQA/ProactiveQA/MultiResponseQA,
                                  # StreamChunk (chunk-wise format), TrainingInstance
  config.py                      # AURADataEngineConfig — N, N', M, chunk size, FPS, etc.
                                  #   (defaults = paper's Section 6.1 values: N=30, N'=15, M=10)
  prompts.py                     # every MLLM/LLM prompt template + rewrite template pools
  llm_client.py                  # MLLMClient interface, MockMLLMClient (offline/deterministic),
                                  # AnthropicMLLMClient (real, frame-sampled Claude calls)
  video_utils.py                 # ffmpeg/ffprobe wrappers + synthetic test-video generator
  stage1_video_preparation.py    # §4.1
  stage2_qa_synthesis.py         # §4.2 (both sub-pipelines)
  stage3_qa_refinement.py        # §4.3
  stage4_streaming_structuring.py# §4.4 (+ §3.1 windowing, §4.6 mixing/acks)
  stage5_quality_verification.py # §4.5
  loss.py                        # §5.1 Silent-Speech Balanced Loss, derived from the
                                  # supervision_mask that Stage 4 already computes
  pipeline.py                    # orchestrates Stage 1 -> 5, collects PipelineStats
run_pipeline.py                  # CLI
tests/test_pipeline_smoke.py     # end-to-end structural-invariant tests (see below)
```

## Quick start

Install as a toolkit — registers the `aura-pipeline` and `aura-viz` commands
(see **[QUICKSTART.md](QUICKSTART.md)** for a guided, bilingual walkthrough):

```bash
pip install -e ".[all]"        # or a minimal install: pip install -r requirements.txt

# Launch the platform (Run pipeline · Browse · Overview) on the bundled demo data:
aura-viz --work-dir real_test/output          # → http://localhost:8000

# Dry run of the pipeline: deterministic mock MLLM, no API key or GPU needed.
aura-pipeline --src example_run/videos --work-dir example_run/output --client mock
# (equivalently: python run_pipeline.py --src ... --client mock)

# Real data generation with Aliyun Bailian (百炼) / DashScope Qwen-VL:
export DASHSCOPE_API_KEY=sk-...        # your Bailian key (or pass --api-key)
python run_pipeline.py --src /path/to/raw_videos --work-dir out/ \
    --client dashscope --model qwen3-vl-plus

# ...or with the Anthropic API:
export ANTHROPIC_API_KEY=sk-...
python run_pipeline.py --src /path/to/raw_videos --work-dir out/ \
    --client anthropic --model claude-sonnet-4-6

# ...or fully OFFLINE with a local open-weights VLM (no cloud key needed).
# Any OpenAI-compatible server works — here Ollama serving Qwen2.5-VL:
#   ollama serve &            # OpenAI-compatible endpoint at :11434/v1
#   ollama pull qwen2.5vl:3b
python run_pipeline.py --src /path/to/raw_videos --work-dir out/ \
    --client ollama --model qwen2.5vl:3b            # or --client openai-compatible --base-url ...
```

**Local / offline VLM (`--client ollama` / `openai-compatible`).** Uses the
standard OpenAI `image_url` frame format against any local server (Ollama,
llama.cpp `llama-server`, LM Studio, vLLM). Two practical notes from a real CPU
run of `qwen2.5vl:3b`: (1) image frames cost ~1000 tokens each, so raise the
server's context above the 4096 default (e.g. an Ollama Modelfile with
`PARAMETER num_ctx 16384`) or calls 400 with `exceed_context_size`; (2) CPU
inference costs tens of seconds *per frame*, so `--max-frames-per-call 3` and a
single short video keep a demo run to minutes, not hours. Set
`NO_PROXY=127.0.0.1,localhost` if you have an `HTTP_PROXY` in your environment,
so the client reaches the local server directly.

**ffmpeg is optional.** Stage 1 uses a system `ffmpeg`/`ffprobe` for the
faithful 2 FPS + H.264 re-encode when they are on `PATH`, and otherwise falls
back to OpenCV (bundled with `opencv-python-headless`) — so the pipeline runs
with no system install. The OpenCV fallback writes an `mp4v`-coded 2 FPS clip
(codec is irrelevant here: it is only re-decoded to JPEG frames for the model),
so install ffmpeg if you specifically need genuine H.264 output on disk.

**Bailian key must be authorized for model *invocation*.** A brand-new key can
often *list* models but not *call* them, returning `403 Model.AccessDenied`. If
you hit that, the client raises a `RuntimeError` spelling out the fix: in the
[Bailian console](https://bailian.console.aliyun.com) complete real-name
verification (实名认证), activate the model service (开通百炼 / the specific
Qwen-VL model), and make sure the key's workspace (业务空间) has the model
authorized with billing/free-tier quota enabled. A plain `qwen-turbo` text call
succeeding is the signal that generation will work. Cheaper VL alternatives:
`qwen3-vl-flash`, `qwen-vl-plus`.

Output: `out/training_instances.jsonl`, one `TrainingInstance` per line — directly
loadable for building the chunk-wise multimodal training sequences (attach a video
frame to every chunk with `text_only: false`, and apply per-token weights from
`loss.py::per_chunk_weights`).

Run the tests:

```bash
python -m pytest tests/ -v
```

I ran this suite (against two synthetic clips, 18s and 75s, with the mock client)
before delivering this; all 9 pass, including:
- video-window span never exceeds `N` seconds in any unrolled instance,
- QA-window history never keeps more than `M` outside groups,
- the supervision mask matches Eq. (1) exactly — every silent turn plus exactly one
  non-silent turn (the target) is marked supervised, never an earlier non-silent turn,
- the ≤N-second clip takes the "full history retained" branch (window starts at t=0),
- Stage 5 actually drops some fraction of samples.

A full run over the two demo clips produced 45 unrolled instances, 38 surviving
Stage-5 quality verification, with **80.1% of supervised turns being silent** —
consistent with the paper's stated motivation for the class-balanced loss (silence
dominates real streaming supervision).

## Configurable VLM agents (`--config`)

Instead of choosing a backend with flags, declare your VLM **agent(s)** in a
config file and — optionally — route the pipeline's distinct sub-tasks to
different models. Each agent is just a configured client; the pipeline is
provider-agnostic.

```bash
python run_pipeline.py --src videos/ --work-dir out/ --config configs/local-qwen.toml
```

A single-agent config (everything uses one model):

```toml
[agents.default]
provider            = "openai_compatible"   # ollama / vLLM / LM Studio / any OpenAI-compatible server
model               = "qwen2.5vl:7b"
base_url            = "http://localhost:11434/v1"
max_frames_per_call = 6
frame_max_side      = 512
```

Per-role routing — e.g. a strong hosted model generates, a cheap local one
verifies (roles: `scene`, `generate`, `verify`, `refine`, `quality`; anything
unlisted uses `default`):

```toml
[roles]
generate = "strong"
verify   = "cheap"

[agents.default]
provider = "openai_compatible"
model    = "qwen2.5vl:3b"
base_url = "http://localhost:11434/v1"

[agents.strong]
provider    = "dashscope"          # Aliyun Bailian
model       = "qwen-vl-max"
api_key_env = "DASHSCOPE_API_KEY"  # key read from env — never hard-code it

[agents.cheap]
provider = "openai_compatible"
model    = "qwen2.5vl:3b"
base_url = "http://localhost:11434/v1"
```

Providers: `openai_compatible` (aliases `ollama`/`vllm`/`lmstudio`/`local`),
`dashscope` (alias `bailian`), `anthropic`, `mock`. Config formats: `.toml` and
`.json` need no extra dependency; `.yaml` needs `pyyaml`. Ready-made configs live
in `configs/` (`local-qwen.toml` for Ollama, `vllm.toml` for a GPU vLLM server,
`bailian.toml`, `mock.toml`, `models.example.toml`). For vLLM, serve on a port
other than the platform's 8000, e.g.:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 \
    --max-model-len 16384 --limit-mm-per-prompt image=16
aura-pipeline --src videos/ --work-dir out/ --config configs/vllm.toml
```

In code:

```python
from aura_data_engine import build_client_from_config, run_full_pipeline, AURADataEngineConfig
client = build_client_from_config("configs/local-qwen.toml")
instances, stats = run_full_pipeline("videos/", "out/", client, AURADataEngineConfig())
```

## Plugging in a real MLLM

Both real backends — `AnthropicMLLMClient` and `DashScopeMLLMClient` (Bailian /
Qwen-VL) — sample up to `max_frames_per_call` frames uniformly from the requested
time range with OpenCV and send them alongside the prompt (chat APIs take images,
not raw video containers). They share every Stage 2/3/5 method through the
`_FrameSampledMLLMClient` base class and differ only in the `_call` primitive, so
adding GPT-4o or a locally-served open-weights Qwen is a one-method subclass.
`DashScopeMLLMClient` sends Qwen-VL a single `video` content block containing the
frame list (better temporal grounding than independent images) and automatically
falls back to per-frame `image_url` blocks for a single frame or if the endpoint
rejects the `video` type. Two things worth tuning for production runs:

1. **Stage 5 is a visual check (§4.5).** `quality_verify_rt` /
   `quality_verify_proactive_multi` sample the retained video window's frames
   (`[window_start, window_end]`) and send them with the QA history, so the judge
   checks *visual* grounding — not text alone. This makes Stage 5 as call-heavy
   as Stage 2 (one vision call per instance); budget for it on real runs.
2. **Cost/latency.** Stage 2's per-candidate verification and Stage 3's
   per-timestamp difficulty-sibling generation are one MLLM call each; at scale
   you'll want batching/async, which `MLLMClient` doesn't preclude but doesn't
   provide either.

## Visualization platform (`aura_viz`)

A dependency-light local web app for inspecting the generated data. A **Browse**
tab lets you filter training instances and see each one as the chunk-wise
streaming timeline with the **actual video frame at every timestamp**, the
Section 5.1 supervision mask, and per-chunk loss weights; an **Overview** tab
gives dataset analytics (instances by QA type, per-video counts, and
distributions of silent-supervision fraction, context size, turns, and
video-window span, all hand-drawn — no charting dependency); a **Run pipeline**
tab executes Stage 1→5 with live progress — including an in-browser **VLM
config editor** (New/Edit next to the config dropdown: start from a template
— local Ollama, Bailian, multi-agent role routing, mock — validate, and save
into `configs/`; saved configs are immediately selectable for the next run).

```bash
python -m aura_viz --work-dir real_test/output --port 8000
# open http://localhost:8000
```

It reads `<work-dir>/training_instances.jsonl` and, when
`<work-dir>/prepared_videos/` is present, extracts frames on demand with OpenCV
(reusing `loss.per_chunk_weights` / `loss.summarize_supervision`, so what you see
matches training exactly). No build step, no extra dependencies beyond
`opencv-python-headless` and the Python standard library. Layout:

```
aura_viz/
  server.py        # stdlib http.server: /api/summary, /api/instances, /api/instance/<id>, /api/frame
  __main__.py      # CLI (--work-dir / --host / --port)
  static/          # index.html + app.css + app.js (vanilla, theme-aware)
```

## Citation

```bibtex
@article{aura2026,
  title={AURA: Always-On Understanding and Real-Time Assistance via Video Streams},
  author={Lu, Xudong and Bo, Yang and Chen, Jinpeng and Li, Shuhan and Guo, Xintong and
          Guan, Huankang and Liu, Fang and Xu, Dunyuan and Sun, Peiwen and Sun, Heyang and
          Liu, Rui and Li, Hongsheng},
  journal={arXiv preprint arXiv:2604.04184},
  year={2026}
}
```
