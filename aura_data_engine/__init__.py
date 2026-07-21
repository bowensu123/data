"""AURA Coarse-to-Fine Streaming Data Engine — reproduction of the data
generation pipeline of arXiv:2604.04184.

Programmatic entry points:

    from aura_data_engine import run_full_pipeline, AURADataEngineConfig
    from aura_data_engine.agents import build_client_from_config

    client = build_client_from_config("configs/local-qwen.toml")   # your VLM agent(s)
    instances, stats = run_full_pipeline("videos/", "out/", client, AURADataEngineConfig())
"""

__version__ = "0.1.0"

from .config import AURADataEngineConfig
from .pipeline import run_full_pipeline, save_instances, PipelineStats
from .agents import (
    AgentSpec, build_agent, RoutedMLLMClient,
    build_client_from_config, build_routed_client_from_dict, load_config_file,
)

__all__ = [
    "__version__",
    "AURADataEngineConfig", "run_full_pipeline", "save_instances", "PipelineStats",
    "AgentSpec", "build_agent", "RoutedMLLMClient",
    "build_client_from_config", "build_routed_client_from_dict", "load_config_file",
]
