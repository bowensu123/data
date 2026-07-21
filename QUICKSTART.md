# QUICKSTART

A toolkit for generating and inspecting **streaming video-LLM training data**
(reproduction of the data pipeline in arXiv:2604.04184). Two things in one:

- **`aura_data_engine`** — the 5-stage data pipeline (Stage 1 video prep → Stage 5
  quality verification), with pluggable, config-driven VLM backends.
- **`aura_viz`** — a local web platform to **run** the pipeline and **visualize**
  the results (per-instance streaming timeline with real frames + dataset analytics).

---

## 中文

### 0. 安装(一次性)
```bash
cd aura_data_engine
pip install -e ".[all]"      # 或: pip install -r requirements.txt
```
`-e ".[all]"` 会装上所有可选后端(openai/anthropic/pyyaml)并注册命令 `aura-viz`、`aura-pipeline`。

### 1. 网页平台(推荐)
```bash
aura-viz --work-dir real_test/output      # 或: python -m aura_viz --work-dir real_test/output
# 打开 http://localhost:8000
```
- **Run pipeline** 页:选源视频目录 + VLM 配置 + 参数 → 点 Run → 看 Stage 1→5 实时进度 → **Load results**
- **Browse** 页:逐条看训练样本(每个时间戳的真实视频帧 + 对话 + 监督掩码 + 损失权重)
- **Overview** 页:数据集统计图表

先用配置 `mock` 秒级试跑一遍(不需要模型)。

### 2. 配置你的 VLM(关键)
编辑 `configs/*.toml`。单模型:
```toml
[agents.default]
provider = "openai_compatible"   # ollama / vLLM / LM Studio / 任意 OpenAI 兼容服务
model    = "qwen2.5vl:7b"
base_url = "http://localhost:11434/v1"
```
百炼(key 从环境变量读):
```bash
export DASHSCOPE_API_KEY=sk-...
```
```toml
[agents.default]
provider = "dashscope"
model    = "qwen-vl-max"
api_key_env = "DASHSCOPE_API_KEY"
```
多模型分工(强模型生成、便宜模型校验):见 `configs/models.example.toml` 的 `[roles]`。

### 3. 本地跑真实 VLM(Ollama)
```bash
ollama serve &            # OpenAI 兼容端口 :11434/v1
ollama pull qwen2.5vl:7b
aura-pipeline --src real_test/videos --work-dir out/ --config configs/local-qwen.toml
```
> 纯 CPU 每次调用要几十秒;先用短视频。想要能通过质量校验的样本,用**强模型 + 内容丰富的长视频**。

### 4. 命令行 / 代码
```bash
aura-pipeline --src videos/ --work-dir out/ --config configs/bailian.toml
```
```python
from aura_data_engine import build_client_from_config, run_full_pipeline, AURADataEngineConfig
client = build_client_from_config("configs/local-qwen.toml")
instances, stats = run_full_pipeline("videos/", "out/", client, AURADataEngineConfig())
```

---

## English

### 0. Install (once)
```bash
cd aura_data_engine
pip install -e ".[all]"      # or: pip install -r requirements.txt
```
Registers the `aura-viz` and `aura-pipeline` commands and all optional backends.

### 1. Web platform (recommended)
```bash
aura-viz --work-dir real_test/output      # open http://localhost:8000
```
Tabs: **Run pipeline** (configure → run Stage 1→5 with live progress → Load results),
**Browse** (per-instance streaming timeline: real frame + text + supervision mask +
loss weights at every second), **Overview** (dataset analytics). Pick the `mock`
config for an instant, model-free run.

### 2. Configure your VLM
Edit `configs/*.toml` — one `[agents.default]` for everything, or add a `[roles]`
table to route `scene`/`generate`/`verify`/`refine`/`quality` to different agents.
Providers: `openai_compatible` (`ollama`/`vllm`/`lmstudio`/`local`), `dashscope`
(`bailian`), `anthropic`, `mock`. Keys come from env via `api_key_env`.

### 3. Real local VLM (Ollama)
```bash
ollama serve & ; ollama pull qwen2.5vl:7b
aura-pipeline --src real_test/videos --work-dir out/ --config configs/local-qwen.toml
```

### 4. CLI / code — see the Chinese section above; same commands.

---

## Output
Each run writes `<work-dir>/training_instances.jsonl` — one `TrainingInstance` per
line (chunk-wise stream + supervision mask + `n_silent_supervised`). Load it in the
platform (`aura-viz --work-dir <work-dir>`) or in your trainer (attach a video frame
to every non-`text_only` chunk; apply `loss.per_chunk_weights`).

## Tests
```bash
pytest -q          # 41 tests
```
