# Comparison with the AURA paper (arXiv:2604.04184)

An honest account of what this project reproduces from the paper's data
pipeline, what it does **not**, and what it adds on top.

> **中文摘要**:算法骨架(五阶段、超参、Silent-Speech 平衡损失)——**高度忠实**;
> 论文的**数据集(~115k 真实样本)、模型训练、以及生产数据用的强 MLLM**——**没有复现**。
> 特别注意:论文**从未具名**它用哪个 MLLM 生产/校验数据(只写 "an MLLM"),
> 所以这块由使用者在 `configs/` 里**自选强模型**。此外本项目还加了论文没有的
> 工程封装(可视化平台、配置化多智能体路由、鲁棒层、工具包)。

## TL;DR

This repo faithfully reproduces the pipeline's **algorithm** (stage structure,
hyperparameters, supervision mask and loss) but not its **dataset**, its
**model training/evaluation**, or the strong generation model the paper relied
on — which the paper never names.

---

## 1. Faithfully reproduced — algorithm & hyperparameters

| Dimension | Paper | This project |
|---|---|---|
| Five-stage structure | video prep → QA synthesis → refinement → streaming structuring → quality verification | one-to-one (`stage1..5`) |
| Hyperparameters | 2 FPS, N=30 s, N′=15 s, M=10 groups, chunk=1 s | identical defaults in `config.py` |
| QA taxonomy | Real-Time / Proactive / Multi-Response with timing invariants | enforced as dataclass invariants in `schema.py` |
| Chunk-wise format + dual sliding-window truncation (§3.1) | described | `stage4_streaming_structuring.py` |
| Silent-Speech Balanced Loss, Eq. (1) | mask = silent + last non-silent; `w_silent = 1/N_silent` | matches exactly in `loss.py` |
| Proactive timing | "the question timestamp **precedes** the answer timestamp" (§4.2) | enforced `a_time > q_time` (strict) |
| RT refinement | 5 candidates, sample 1 (balanced), then **generate an answer for the sampled question** (§4.3) | `stage3` re-generates for whichever question is sampled |
| Multi-Response answers | "multiple valid answers at **different timestamps**" (§4.2) | distinct-timestamp dedup + ≥2 required |
| Stage 5 verification | judge against **visual evidence / retained video context** + QA history (§4.5) | samples the retained window's frames `[window_start, window_end]` and sends them to the judge |
| Acknowledgments + timestamp-mixing of the three QA types (§4.6) | described | implemented |
| Fully automated, no human-in-the-loop | yes | yes |

Every algorithmic rule the paper *specifies* is now aligned to the paper (the
last four rows were tightened on 2026-07-22 to remove earlier judgment-call
relaxations). Per-stage choices where the paper is genuinely silent are
documented in the README table.

## 2. Key differences — what we did NOT (or cannot) reproduce

> The **algorithm** now matches the paper wherever it is specified (§1). The
> differences below are things the paper does not publish, or that need
> resources beyond a data-pipeline reproduction — **not** algorithmic
> divergences.


1. **The dataset itself (the biggest gap).** Paper: **~115k streaming QA samples
   / 1.04B tokens** (plus ~59k offline / 0.16B) from a broad multi-domain internet
   corpus (sports, vlogs, documentaries, TV, movies, courses, games, animation).
   Here: with the models we had access to (mock + local Qwen2.5-VL 3B/7B) the
   number of **verification-passing real instances was 0** (small models fail
   their own Stage-2 verification), plus a 639-instance **mock** demo (real
   structure, synthetic text). We have the machinery, not the corpus — and no
   domain curation (the paper's Figure 5 distribution).

2. **The generation MLLM.** Paper uses an unnamed (presumably strong) MLLM for
   scene segmentation / QA generation / verification. We only ran mock and small
   local Qwen-VL, which are a different quality tier. See §3.

3. **Prompts and rewrite templates.** The paper **does not publish** its prompts,
   its "predefined candidate templates", or its verification rubric. Ours are
   reconstructed from the algorithmic description — functionally aligned, but not
   identical, which is also why their filtering pass-rates can't be matched.

4. **No training / evaluation.** Paper fine-tunes **Qwen3-VL-8B-Instruct** (LLM
   component only; vision encoder + connector frozen; global batch 128, lr 1e-5,
   1 epoch) and benchmarks it. This project **only produces data + a reference
   loss implementation** — there is no training loop, no trained model, no eval.

5. **Streaming only.** The paper also mixes in ~59k **offline** video-QA samples;
   we implement only the streaming pipeline.

6. **Video preparation.** Paper: 2 FPS + true H.264 re-encode. Here: identical
   when a system `ffmpeg` is present; otherwise an OpenCV `mp4v` fallback
   (frame-sampling-equivalent, not byte-identical encoding).

7. **Scale/engineering.** No batched/async MLLM calls or distributed processing —
   this is a reference-scale reproduction, not a 174k-sample production run.

## 3. The data-generation MLLM is NOT named in the paper

Across §4.2–4.5 the generation/verification model is referred to only
generically:

> "we first apply **a multimodal large language model (MLLM)** to perform scene
> segmentation…"

It is never named — not in the body, appendix, acknowledgements, or footnotes.
Do not confuse it with the model being *trained*:

| Role | Model | Named in paper? |
|---|---|---|
| **Generates** the data (runs the pipeline) | unknown | ❌ no |
| **Is trained** on the data (the target) | Qwen3-VL-8B-Instruct | ✅ yes |

**Speculation (not stated by the paper):** data-generation "teacher" models are
usually *stronger* than the training target (8B), and both generation and
self-verification demand high capability — so it was likely a larger/stronger
VLM (e.g. a GPT-4o / Gemini / Qwen-VL-Max / Qwen2.5-VL-72B class model). But this
is a guess; the paper gives no model id.

**Implication for reproduction:** because the paper leaves this open, the choice
is yours — which is exactly why the backend is config-driven. To approach the
paper's data quality, point `configs/` at a **strong VLM** (e.g. Bailian
`qwen-vl-max`, a large local Qwen-VL, or a commercial API), not a local 3B/7B.
The pipeline is unchanged; only the config differs.

## 4. What this project adds beyond the paper

The paper's released repo ships the **model + a real-time inference framework**
(vLLM serving, ASR, TTS, context management) and does **not** release the data
pipeline. This project reconstructs the data pipeline from scratch and adds
tooling the paper does not describe:

- **`aura_viz`** — a local web platform to run the full pipeline (live progress)
  and visualize instances (real frames + supervision mask) + dataset analytics.
- **Config-driven multi-agent VLM routing** — declare agents and route roles
  (scene / generate / verify / refine / quality) to different models.
- **Robustness layer** (`parse_utils`, RT normalization, skip-bad-item) — needed
  because we used weak models; a strong MLLM presumably would not require it.
- **Toolkit packaging** (`pyproject`, console commands), a 20-clip sample video
  set, and an offline demo kit.

---

**One line:** the *how* (algorithm, hyperparameters, loss) is faithfully
reproduced; the *what it produced* (115k real samples, a strong generation MLLM,
a trained model, benchmarks) is not — and the paper never tells you which MLLM
generated its data.
