"""Tests for the configurable VLM agent layer (agents.py)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_data_engine.agents import (
    AgentSpec, build_agent, RoutedMLLMClient, ROLE_OF_METHOD,
    build_routed_client_from_dict, build_client_from_config, validate_config_dict,
)
from aura_data_engine.llm_client import MockMLLMClient, OpenAICompatibleMLLMClient


def tagged(tag):
    c = MockMLLMClient(seed=0)
    c.tag = tag
    return c


class TestBuildAgent(unittest.TestCase):
    def test_mock(self):
        c = build_agent(AgentSpec(name="d", provider="mock", seed=1, pass_rate=0.5))
        self.assertIsInstance(c, MockMLLMClient)

    def test_openai_compatible_needs_model(self):
        with self.assertRaises(ValueError):
            build_agent(AgentSpec(name="d", provider="ollama"))  # no model

    def test_openai_compatible_ok(self):
        c = build_agent(AgentSpec(name="d", provider="ollama", model="qwen2.5vl:7b",
                                  base_url="http://localhost:11434/v1", frame_max_side=400))
        self.assertIsInstance(c, OpenAICompatibleMLLMClient)
        self.assertEqual(c.frame_max_side, 400)

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            build_agent(AgentSpec(name="d", provider="not-a-provider", model="x"))

    def test_unknown_option_rejected(self):
        with self.assertRaises(ValueError):
            AgentSpec.from_dict("d", {"provider": "mock", "temperatur": 0.2})  # typo


class TestRouting(unittest.TestCase):
    def test_single_agent_routes_everything_to_default(self):
        r = RoutedMLLMClient({"default": tagged("D")})
        for method in ROLE_OF_METHOD:
            self.assertEqual(r.agent_for(method).tag, "D")

    def test_per_role_routing(self):
        r = RoutedMLLMClient(
            {"default": tagged("D"), "cheap": tagged("C"), "strong": tagged("S")},
            roles={"verify": "cheap", "generate": "strong"})
        self.assertEqual(r.agent_for("verify_rt_qa").tag, "C")          # verify role
        self.assertEqual(r.agent_for("check_multi_answerable").tag, "C")
        self.assertEqual(r.agent_for("generate_candidate_qas").tag, "S")  # generate role
        self.assertEqual(r.agent_for("segment_scenes").tag, "D")        # scene -> default
        self.assertEqual(r.agent_for("rewrite_question").tag, "D")      # refine -> default

    def test_missing_default_raises(self):
        with self.assertRaises(ValueError):
            RoutedMLLMClient({"cheap": tagged("C")}, roles={})

    def test_unknown_role_agent_raises(self):
        with self.assertRaises(ValueError):
            RoutedMLLMClient({"default": tagged("D")}, roles={"verify": "ghost"})

    def test_dispatch_actually_calls_routed_agent(self):
        cheap = tagged("C")
        cheap.verify_rt_qa = lambda *a, **k: {"which": "cheap"}
        default = tagged("D")
        default.verify_rt_qa = lambda *a, **k: {"which": "default"}
        r = RoutedMLLMClient({"default": default, "cheap": cheap}, roles={"verify": "cheap"})
        self.assertEqual(r.verify_rt_qa("v.mp4", "q", "a", 0.0)["which"], "cheap")
        # a non-verify call still hits default
        self.assertEqual(r.agent_for("generate_candidate_qas").tag, "D")

    def test_describe(self):
        r = RoutedMLLMClient({"default": tagged("D"), "cheap": tagged("C")}, roles={"verify": "cheap"})
        d = r.describe()
        self.assertEqual(d["verify"], "cheap")
        self.assertEqual(d["generate"], "default")


class TestConfigDict(unittest.TestCase):
    def test_flat_single_agent(self):
        r = build_routed_client_from_dict({"provider": "mock", "seed": 3})
        self.assertIsInstance(r.agents["default"], MockMLLMClient)

    def test_agents_table_with_roles(self):
        cfg = {
            "agents": {
                "default": {"provider": "mock", "seed": 0},
                "cheap": {"provider": "mock", "seed": 1},
            },
            "roles": {"verify": "cheap"},
        }
        r = build_routed_client_from_dict(cfg)
        self.assertEqual(r.roles["verify"], "cheap")
        self.assertIn("default", r.agents)

    def test_single_named_agent_becomes_default(self):
        r = build_routed_client_from_dict({"agents": {"only": {"provider": "mock"}}})
        self.assertIn("default", r.agents)

    def test_toml_file_roundtrip(self):
        toml = ('[agents.default]\nprovider = "mock"\nseed = 7\n'
                '[agents.cheap]\nprovider = "mock"\nseed = 9\n'
                '[roles]\nverify = "cheap"\n')
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
            f.write(toml); path = f.name
        try:
            r = build_client_from_config(path)
            self.assertEqual(r.agent_for("verify_rt_qa"), r.agents["cheap"])
            self.assertEqual(r.agent_for("segment_scenes"), r.agents["default"])
        finally:
            os.unlink(path)


class TestValidateConfigDict(unittest.TestCase):
    def test_valid_single_agent(self):
        cfg = {"agents": {"default": {"provider": "openai_compatible", "model": "m",
                                       "base_url": "http://x/v1"}}}
        self.assertEqual(validate_config_dict(cfg), [])

    def test_valid_routed(self):
        cfg = {"agents": {"default": {"provider": "mock"}, "cheap": {"provider": "mock"}},
               "roles": {"verify": "cheap"}}
        self.assertEqual(validate_config_dict(cfg), [])

    def test_unknown_provider_reported(self):
        cfg = {"agents": {"default": {"provider": "no-such"}}}
        problems = validate_config_dict(cfg)
        self.assertTrue(any("unknown provider" in p for p in problems))

    def test_missing_model_for_local(self):
        cfg = {"agents": {"default": {"provider": "ollama"}}}
        problems = validate_config_dict(cfg)
        self.assertTrue(any("'model' is required" in p for p in problems))

    def test_unknown_role_and_agent(self):
        cfg = {"agents": {"default": {"provider": "mock"}},
               "roles": {"nonsense": "default", "verify": "ghost"}}
        problems = validate_config_dict(cfg)
        self.assertTrue(any("unknown role 'nonsense'" in p for p in problems))
        self.assertTrue(any("unknown agent 'ghost'" in p for p in problems))

    def test_typo_field_reported(self):
        cfg = {"agents": {"default": {"provider": "mock", "temperatur": 0.2}}}
        problems = validate_config_dict(cfg)
        self.assertTrue(any("unknown option" in p for p in problems))

    def test_empty_config(self):
        self.assertTrue(validate_config_dict({}))  # non-empty problem list

    def test_does_not_require_api_key(self):
        # validation is static: a dashscope agent without a key in the env is fine
        cfg = {"agents": {"default": {"provider": "dashscope", "model": "qwen-vl-max",
                                       "api_key_env": "SOME_UNSET_VAR"}}}
        self.assertEqual(validate_config_dict(cfg), [])


if __name__ == "__main__":
    unittest.main()
