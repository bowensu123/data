# Source of the real test videos in this folder

`videos/bottle-detection.mp4` and `videos/car-detection.mp4` are pulled from
[`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos),
licensed under **CC-BY 4.0**. They're short (30–40s), real (non-synthetic) clips,
used here purely to exercise the pipeline end-to-end with real footage instead of
the ffmpeg-generated `testsrc` pattern used by the unit tests.

Downloaded directly via `raw.githubusercontent.com` (an allowed egress domain in
this sandbox); general video-hosting sites (YouTube etc.) are not reachable here.

## What this run does and does not validate

- **Does validate**: Stage 1 (`ffprobe`/`ffmpeg` resample to 2 FPS + H.264
  re-encode) against real, variable-native-fps source video — confirmed both clips
  came out at exactly 2.0 FPS / H.264 afterward. Also confirms `AnthropicMLLMClient`'s
  OpenCV frame-sampling path can read real frames from the prepared output.
- **Does NOT validate**: the semantic quality of QA content, because Stages 2/3/5
  ran with `MockMLLMClient` — there is no `ANTHROPIC_API_KEY` in this sandbox and
  general internet access to any other MLLM provider is blocked, so no real model
  ever looked at these frames. The questions/answers in `output/training_instances.jsonl`
  are synthetic placeholders; only the *pipeline mechanics* (chunking, windowing,
  unrolling, masking, filtering) are meaningfully exercised here.

To get real content, run this same command from a machine with API access —
e.g. with Aliyun Bailian (百炼) / Qwen-VL:

```bash
export DASHSCOPE_API_KEY=sk-...
python run_pipeline.py --src real_test/videos --work-dir real_test/output_real \
    --client dashscope --model qwen3-vl-plus
```

or with the Anthropic API:

```bash
export ANTHROPIC_API_KEY=sk-...
python run_pipeline.py --src real_test/videos --work-dir real_test/output_real \
    --client anthropic --model claude-sonnet-4-6
```

Note: the Bailian key used on 2026-07-20 returned `403 Model.AccessDenied` for
every model (it could list models but not invoke them), so this folder's
`output/` is still the mock-content run. Once the key is authorized for model
invocation (see the README's "Bailian key must be authorized" note), the
command above produces real Qwen-VL content.
