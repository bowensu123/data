#!/usr/bin/env python3
"""
`aura-pipeline` — run the AURA Coarse-to-Fine Streaming Data Engine from the
command line. (Reproduction of the data pipeline of arXiv:2604.04184.)

Examples
--------
Offline dry run (deterministic mock MLLM — no API key or GPU):

    aura-pipeline --src videos/ --work-dir out/ --client mock

Config-driven run — declare your VLM agent(s) in a file (recommended):

    aura-pipeline --src videos/ --work-dir out/ --config configs/local-qwen.toml

Or pick a backend directly with flags:

    export DASHSCOPE_API_KEY=sk-...
    aura-pipeline --src videos/ --work-dir out/ --client dashscope --model qwen-vl-max

(`python run_pipeline.py ...` is a thin wrapper around this and works too.)
"""

import argparse
import json
import logging
import sys

from .config import AURADataEngineConfig
from .llm_client import (
    MockMLLMClient, AnthropicMLLMClient, DashScopeMLLMClient,
    OpenAICompatibleMLLMClient, DASHSCOPE_DEFAULT_MODEL,
)
from .agents import build_client_from_config
from .loss import summarize_supervision
from .pipeline import run_full_pipeline, save_instances


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aura-pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Directory of raw source videos")
    ap.add_argument("--work-dir", required=True, help="Directory for intermediate + output artifacts")
    ap.add_argument("--config", default=None,
                    help="Agent config file (.toml/.json/.yaml) declaring your VLM agent(s) and "
                         "optional per-role routing. Overrides --client/--model. See configs/.")
    ap.add_argument("--client",
                    choices=["mock", "anthropic", "dashscope", "bailian", "ollama", "openai-compatible"],
                    default="mock",
                    help="MLLM backend when --config is not given. 'bailian' aliases 'dashscope'; "
                         "'ollama'/'openai-compatible' target a local OpenAI-compatible VLM server.")
    ap.add_argument("--model", default=None,
                    help="Model name. Defaults per client: claude-sonnet-4-6 (anthropic), "
                         f"{DASHSCOPE_DEFAULT_MODEL} (dashscope/bailian), qwen2.5vl (ollama).")
    ap.add_argument("--api-key", default=None,
                    help="API key for the dashscope/bailian client (else DASHSCOPE_API_KEY / "
                         "BAILIAN_API_KEY env var). Ignored by local backends.")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL for --client ollama/openai-compatible "
                         "(default http://localhost:11434/v1 for ollama).")
    ap.add_argument("--max-frames-per-call", type=int, default=None,
                    help="Frames sampled per MLLM call (default 16 cloud, 8 local — fewer = faster on CPU)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--video-window-n", type=int, default=30, help="N seconds (Section 6.1)")
    ap.add_argument("--prefix-margin-n-prime", type=int, default=15, help="N' seconds (Section 6.1)")
    ap.add_argument("--qa-window-m", type=int, default=10, help="M QA groups (Section 6.1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = AURADataEngineConfig(
        target_fps=args.fps,
        video_window_n_s=args.video_window_n,
        prefix_margin_n_prime_s=args.prefix_margin_n_prime,
        qa_window_m_groups=args.qa_window_m,
        random_seed=args.seed,
    )

    # Only forward --max-frames-per-call when set, so each client's own default applies.
    frames_kw = {} if args.max_frames_per_call is None else {"max_frames_per_call": args.max_frames_per_call}

    if args.config:
        client = build_client_from_config(args.config)
        logging.getLogger("aura_data_engine").info(
            "loaded agent config %s; role routing: %s", args.config, client.describe())
    elif args.client == "mock":
        client = MockMLLMClient(seed=args.seed)
    elif args.client == "anthropic":
        client = AnthropicMLLMClient(model=args.model or "claude-sonnet-4-6", **frames_kw)
    elif args.client in ("dashscope", "bailian"):
        client = DashScopeMLLMClient(
            model=args.model or DASHSCOPE_DEFAULT_MODEL, api_key=args.api_key, **frames_kw)
    else:  # "ollama" / "openai-compatible" — local OpenAI-compatible VLM
        client = OpenAICompatibleMLLMClient(
            model=args.model or "qwen2.5vl",
            base_url=args.base_url or "http://localhost:11434/v1", **frames_kw)

    instances, stats = run_full_pipeline(args.src, args.work_dir, client, cfg)

    out_path = f"{args.work_dir.rstrip('/')}/training_instances.jsonl"
    save_instances(instances, out_path)

    print(json.dumps({
        "pipeline_stats": stats.as_dict(),
        "supervision_summary": summarize_supervision(instances),
        "n_final_training_instances": len(instances),
        "output_path": out_path,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
