"""
Configurable VLM "agents".

Instead of picking a backend with CLI flags, you declare one or more *model
agents* in a config file (TOML / JSON / YAML) and — optionally — route the
pipeline's distinct sub-tasks to different agents. Each agent is just a
configured `MLLMClient`; the pipeline never needs to know which provider or
model it is.

Minimal config (one model for everything), `models.toml`:

    [agents.default]
    provider = "openai_compatible"      # ollama / vLLM / LM Studio / any OpenAI-compatible server
    model    = "qwen2.5vl:7b"
    base_url = "http://localhost:11434/v1"
    max_frames_per_call = 6
    frame_max_side = 512

Per-role routing (strong model generates, cheap model verifies):

    [roles]
    generate = "strong"
    verify   = "cheap"

    [agents.default]
    provider = "openai_compatible"
    model    = "qwen2.5vl:3b"
    base_url = "http://localhost:11434/v1"

    [agents.strong]
    provider    = "dashscope"           # Aliyun Bailian
    model       = "qwen-vl-max"
    api_key_env = "DASHSCOPE_API_KEY"

    [agents.cheap]
    provider = "openai_compatible"
    model    = "qwen2.5vl:3b"
    base_url = "http://localhost:11434/v1"

Then:  python run_pipeline.py --src videos/ --work-dir out/ --config models.toml
Or in code:  client = build_client_from_config("models.toml")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import (
    MLLMClient, MockMLLMClient, AnthropicMLLMClient, DashScopeMLLMClient,
    OpenAICompatibleMLLMClient,
)

# Which pipeline sub-task ("role") each MLLMClient method belongs to. A config's
# [roles] table maps these role names to agent names; anything unmapped uses the
# "default" agent.
ROLE_OF_METHOD: Dict[str, str] = {
    "segment_scenes": "scene",
    "generate_candidate_qas": "generate",
    "generate_multi_candidate_questions": "generate",
    "generate_multi_answers": "generate",
    "generate_difficulty_siblings": "generate",
    "generate_answer_for_question": "generate",
    "verify_rt_qa": "verify",
    "verify_proactive_qa": "verify",
    "check_multi_answerable": "verify",
    "verify_multi_answer": "verify",
    "rewrite_question": "refine",
    "quality_verify_rt": "quality",
    "quality_verify_proactive_multi": "quality",
}

# provider aliases -> canonical
_PROVIDER_ALIASES = {
    "mock": "mock",
    "anthropic": "anthropic", "claude": "anthropic",
    "dashscope": "dashscope", "bailian": "dashscope", "qwen": "dashscope", "aliyun": "dashscope",
    "openai_compatible": "openai_compatible", "openai-compatible": "openai_compatible",
    "ollama": "openai_compatible", "openai": "openai_compatible", "local": "openai_compatible",
    "vllm": "openai_compatible", "lmstudio": "openai_compatible", "llamacpp": "openai_compatible",
}


@dataclass
class AgentSpec:
    """Declarative description of one VLM agent (a configured MLLMClient)."""
    name: str = "default"
    provider: str = "openai_compatible"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    max_frames_per_call: Optional[int] = None
    frame_max_side: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_retries: Optional[int] = None
    request_timeout: Optional[float] = None
    frame_format: Optional[str] = None
    seed: Optional[int] = None          # mock only
    pass_rate: Optional[float] = None   # mock only

    @staticmethod
    def from_dict(name: str, d: Dict[str, Any]) -> "AgentSpec":
        known = AgentSpec.__dataclass_fields__.keys()
        unknown = [k for k in d if k not in known and k != "name"]
        if unknown:
            raise ValueError(f"agent '{name}': unknown option(s) {unknown}. "
                             f"Allowed: {sorted(k for k in known if k != 'name')}")
        return AgentSpec(name=name, **{k: v for k, v in d.items() if k != "name"})

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


def _kw(**pairs) -> Dict[str, Any]:
    """Drop None values so each client's own defaults apply."""
    return {k: v for k, v in pairs.items() if v is not None}


def build_agent(spec: AgentSpec) -> MLLMClient:
    """Instantiate the MLLMClient described by `spec`."""
    provider = _PROVIDER_ALIASES.get(spec.provider.lower())
    if provider is None:
        raise ValueError(f"agent '{spec.name}': unknown provider '{spec.provider}'. "
                         f"Use one of: mock, anthropic, dashscope/bailian, openai_compatible/ollama.")

    if provider == "mock":
        return MockMLLMClient(**_kw(seed=spec.seed, pass_rate=spec.pass_rate))

    if provider == "anthropic":
        return AnthropicMLLMClient(**_kw(
            model=spec.model, max_frames_per_call=spec.max_frames_per_call, max_tokens=spec.max_tokens))

    if provider == "dashscope":
        return DashScopeMLLMClient(**_kw(
            model=spec.model, api_key=spec.resolved_api_key(), base_url=spec.base_url,
            max_frames_per_call=spec.max_frames_per_call, frame_max_side=spec.frame_max_side,
            max_tokens=spec.max_tokens, temperature=spec.temperature, max_retries=spec.max_retries,
            request_timeout=spec.request_timeout, frame_format=spec.frame_format))

    # openai_compatible
    if not spec.model:
        raise ValueError(f"agent '{spec.name}': 'model' is required for provider '{spec.provider}'.")
    return OpenAICompatibleMLLMClient(**_kw(
        model=spec.model, base_url=spec.base_url, api_key=spec.resolved_api_key() or "local",
        max_frames_per_call=spec.max_frames_per_call, frame_max_side=spec.frame_max_side,
        max_tokens=spec.max_tokens, temperature=spec.temperature, max_retries=spec.max_retries,
        request_timeout=spec.request_timeout, frame_format=spec.frame_format))


class RoutedMLLMClient(MLLMClient):
    """
    An MLLMClient that dispatches each interface method to a per-role agent.

    `agents` maps agent-name -> MLLMClient (must include "default"); `roles`
    maps a role name (see ROLE_OF_METHOD) -> agent-name. Any role not listed
    falls back to the "default" agent, so a single-agent config routes
    everything to one model.
    """

    def __init__(self, agents: Dict[str, MLLMClient], roles: Optional[Dict[str, str]] = None,
                 default: str = "default"):
        if default not in agents:
            raise ValueError(f"no '{default}' agent defined (agents: {sorted(agents)})")
        roles = roles or {}
        for role, agent_name in roles.items():
            if agent_name not in agents:
                raise ValueError(f"role '{role}' -> unknown agent '{agent_name}' "
                                 f"(agents: {sorted(agents)})")
        self.agents = agents
        self.roles = roles
        self.default = default

    def agent_for(self, method_name: str) -> MLLMClient:
        role = ROLE_OF_METHOD.get(method_name, "")
        return self.agents[self.roles.get(role, self.default)]

    def describe(self) -> Dict[str, str]:
        """role -> agent-name it resolves to (for logging/inspection)."""
        out = {"default": self.default}
        for role in sorted(set(ROLE_OF_METHOD.values())):
            out[role] = self.roles.get(role, self.default)
        return out

    # ---- interface: every method just forwards to its role's agent ----
    def segment_scenes(self, *a, **k):
        return self.agent_for("segment_scenes").segment_scenes(*a, **k)

    def generate_candidate_qas(self, *a, **k):
        return self.agent_for("generate_candidate_qas").generate_candidate_qas(*a, **k)

    def verify_rt_qa(self, *a, **k):
        return self.agent_for("verify_rt_qa").verify_rt_qa(*a, **k)

    def verify_proactive_qa(self, *a, **k):
        return self.agent_for("verify_proactive_qa").verify_proactive_qa(*a, **k)

    def generate_multi_candidate_questions(self, *a, **k):
        return self.agent_for("generate_multi_candidate_questions").generate_multi_candidate_questions(*a, **k)

    def check_multi_answerable(self, *a, **k):
        return self.agent_for("check_multi_answerable").check_multi_answerable(*a, **k)

    def generate_multi_answers(self, *a, **k):
        return self.agent_for("generate_multi_answers").generate_multi_answers(*a, **k)

    def verify_multi_answer(self, *a, **k):
        return self.agent_for("verify_multi_answer").verify_multi_answer(*a, **k)

    def generate_difficulty_siblings(self, *a, **k):
        return self.agent_for("generate_difficulty_siblings").generate_difficulty_siblings(*a, **k)

    def generate_answer_for_question(self, *a, **k):
        return self.agent_for("generate_answer_for_question").generate_answer_for_question(*a, **k)

    def rewrite_question(self, *a, **k):
        return self.agent_for("rewrite_question").rewrite_question(*a, **k)

    def quality_verify_rt(self, *a, **k):
        return self.agent_for("quality_verify_rt").quality_verify_rt(*a, **k)

    def quality_verify_proactive_multi(self, *a, **k):
        return self.agent_for("quality_verify_proactive_multi").quality_verify_proactive_multi(*a, **k)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_config_file(path: str) -> Dict[str, Any]:
    """Load a TOML / JSON / YAML config into a dict. TOML and JSON need no extra
    dependency (stdlib); YAML requires pyyaml."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        raw = f.read()
    if ext in (".toml", ".tml"):
        import tomllib  # Python 3.11+
        return tomllib.loads(raw.decode("utf-8"))
    if ext in (".json",):
        return json.loads(raw.decode("utf-8"))
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # optional
        except ImportError as e:
            raise ImportError("YAML config needs pyyaml: pip install pyyaml "
                              "(or use a .toml / .json config instead)") from e
        return yaml.safe_load(raw.decode("utf-8"))
    raise ValueError(f"unsupported config extension '{ext}' (use .toml, .json, or .yaml)")


KNOWN_ROLES = sorted(set(ROLE_OF_METHOD.values()))


def validate_config_dict(config: Dict[str, Any]) -> List[str]:
    """Statically validate an agent config dict WITHOUT instantiating clients
    (no SDK imports, no API-key checks — those depend on the run environment).

    Returns a list of human-readable problems; empty list = valid. Mirrors the
    normalization rules of `build_routed_client_from_dict`.
    """
    problems: List[str] = []
    if not isinstance(config, dict):
        return ["config root must be a table/object"]
    agents_cfg = config.get("agents")
    roles = config.get("roles", {}) or {}
    if not isinstance(roles, dict):
        problems.append("[roles] must be a table of role = \"agent-name\" pairs")
        roles = {}
    if not agents_cfg:
        flat = {k: v for k, v in config.items() if k != "roles"}
        if "provider" not in flat:
            return problems + ["no [agents] table and no top-level 'provider' key"]
        agents_cfg = {"default": flat}
    if not isinstance(agents_cfg, dict) or not agents_cfg:
        return problems + ["[agents] must be a non-empty table of agent tables"]
    if "default" not in agents_cfg and len(agents_cfg) > 1:
        problems.append(f"multiple agents defined but none named 'default': {sorted(agents_cfg)}")

    for name, d in agents_cfg.items():
        if not isinstance(d, dict):
            problems.append(f"agent '{name}' must be a table ([agents.{name}])")
            continue
        try:
            spec = AgentSpec.from_dict(name, d)
        except (ValueError, TypeError) as e:
            problems.append(str(e))
            continue
        provider = _PROVIDER_ALIASES.get(str(spec.provider).lower())
        if provider is None:
            problems.append(f"agent '{name}': unknown provider '{spec.provider}' "
                            f"(use mock / anthropic / dashscope / openai_compatible)")
        elif provider == "openai_compatible" and not spec.model:
            problems.append(f"agent '{name}': 'model' is required for provider '{spec.provider}'")
        if spec.frame_format not in (None, "video", "image_url"):
            problems.append(f"agent '{name}': frame_format must be 'video' or 'image_url'")

    # After single-agent normalization the only agent becomes "default".
    effective = set(agents_cfg) if "default" in agents_cfg or len(agents_cfg) > 1 \
        else {"default"}
    for role, aname in roles.items():
        if role not in KNOWN_ROLES:
            problems.append(f"unknown role '{role}' (valid roles: {KNOWN_ROLES})")
        if aname not in effective:
            problems.append(f"role '{role}' -> unknown agent '{aname}' (agents: {sorted(effective)})")
    return problems


def build_routed_client_from_dict(config: Dict[str, Any]) -> RoutedMLLMClient:
    """Build a RoutedMLLMClient from a parsed config dict.

    Expected shape:
        {"agents": {"default": {...}, "strong": {...}}, "roles": {"verify": "cheap"}}
    A bare single-agent dict (no "agents" key) is also accepted and treated as
    the default agent.
    """
    agents_cfg = config.get("agents")
    roles = config.get("roles", {}) or {}
    if not agents_cfg:
        # allow a flat single-agent config: the whole dict is one agent
        flat = {k: v for k, v in config.items() if k not in ("roles",)}
        if "provider" not in flat:
            raise ValueError("config has no [agents] table and no 'provider' key.")
        agents_cfg = {"default": flat}
    if "default" not in agents_cfg:
        if len(agents_cfg) == 1:
            # a single named agent becomes the default
            only = next(iter(agents_cfg))
            agents_cfg = {"default": agents_cfg[only]}
        else:
            raise ValueError(f"multiple agents defined but none named 'default': {sorted(agents_cfg)}")

    built: Dict[str, MLLMClient] = {}
    for name, spec_dict in agents_cfg.items():
        built[name] = build_agent(AgentSpec.from_dict(name, spec_dict or {}))
    return RoutedMLLMClient(built, roles=roles, default="default")


def build_client_from_config(path: str) -> RoutedMLLMClient:
    """Load a config file and build the routed multi-agent client from it."""
    return build_routed_client_from_dict(load_config_file(path))
