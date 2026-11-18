# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for exact token-ID requests and YAML-to-AISBench integration."""

import ast
import asyncio
import importlib.util
import json
import os
import re
import shlex
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("fixed_input", ROOT / "tools/fixed_input.py")
fixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed)


@pytest.mark.parametrize("value", [0, -1, True, False, None, "2048", 1.5])
def test_invalid_fixed_length(value):
    with pytest.raises(ValueError, match="positive integer"):
        fixed.positive_token_count(value)


@pytest.mark.parametrize(
    "ids,padding,count,expected",
    [
        ([9, 8], [1, 2], 5, [1, 2, 1, 9, 8]),
        ([9, 8], [], 2, [9, 8]),
        ([7, 9, 8], [], 2, [9, 8]),
        ([], [3], 3, [3, 3, 3]),
    ],
)
def test_fit_prompt_tokens(ids, padding, count, expected):
    assert fixed.fit_prompt_tokens(ids, padding, count) == expected


@pytest.mark.parametrize("ids,padding", [([True], [1]), ([-1], [1]), (["1"], [1]), ([], []), ([], [False])])
def test_invalid_token_ids(ids, padding):
    with pytest.raises(ValueError):
        fixed.fit_prompt_tokens(ids, padding, 4)


def test_fixed_length_config_is_opt_in_and_performance_only():
    assert fixed.fixed_input_options({"case_type": "accuracy"}) == (None, "auto")
    config = dict(case_type="performance", request_conf="vllm_api_general_stream", fixed_input_tokens=2048)
    assert fixed.fixed_input_options(config) == (2048, "auto")
    with pytest.raises(ValueError, match="only for performance"):
        fixed.fixed_input_options({**config, "case_type": "accuracy"})
    with pytest.raises(ValueError, match="completions"):
        fixed.fixed_input_options({**config, "request_conf": "vllm_api_stream_chat"})
    with pytest.raises(ValueError, match="nonempty"):
        fixed.fixed_input_options({**config, "tokenizer_mode": ""})


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 2047}, {"prompt_tokens": "2048"}, {"prompt_tokens": True}])
def test_mismatched_or_missing_usage_is_rejected(usage):
    with pytest.raises(ValueError, match="Fixed input length mismatch"):
        fixed.validate_prompt_usage(usage, 2048)


@pytest.fixture
def backend(monkeypatch):
    class Base:
        def __init__(self, **kwargs):
            self.stream = kwargs.get("stream", True)
            self.response = {"usage": {"prompt_tokens": 8, "completion_tokens": 3}}

        async def get_request_body(self, input_data, max_out_len, output, **kwargs):
            return dict(prompt=input_data, stream=self.stream, max_tokens=max_out_len)

        async def _parse_usage(self, data, output):
            if data.get("usage"):
                output.input_tokens = data["usage"].get("prompt_tokens", 0)

        async def stream_infer(self, request_body, output):
            await self._parse_usage(self.response, output)
            output.success = True

        text_infer = stream_infer

    package = ModuleType("ais_bench.benchmark.models")
    package.VLLMCustomAPI = Base
    monkeypatch.setitem(sys.modules, "ais_bench.benchmark.models", package)
    monkeypatch.setitem(sys.modules, "tools.fixed_input", fixed)
    spec = importlib.util.spec_from_file_location("tools.aisbench_fixed_input", ROOT / "tools/aisbench_fixed_input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "tools.aisbench_fixed_input", module)
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [10, 11] if text == fixed.PADDING_TEXT else [20, 21, 22])
    monkeypatch.setattr(module, "load_fixed_input_tokenizer", lambda *args: tokenizer)
    return module


@pytest.mark.parametrize("stream", [True, False])
def test_backend_sends_ids_and_validates_completed_response(backend, stream):
    model = backend.FixedInputVLLMAPI(path="/model", fixed_input_tokens=8, stream=stream)
    output = SimpleNamespace(extra_details_data={}, success=False)
    body = asyncio.run(model.get_request_body("question", 3, output))
    assert body["prompt"] == [10, 11, 10, 11, 10, 20, 21, 22]
    assert body["add_special_tokens"] is False
    assert body["max_tokens"] == 3
    if stream:
        assert body["stream_options"]["include_usage"] is True
    infer = model.stream_infer if stream else model.text_infer
    asyncio.run(infer(body, output))
    assert output.input_tokens == 8
    assert output.extra_details_data["fixed_input"]["final_input_token_ids"] == body["prompt"]
    assert output.extra_details_data["fixed_input"]["usage_validated"]
    model.response = {}
    with pytest.raises(ValueError, match="missing usage"):
        asyncio.run(infer(body, output))
    assert output.success is False
    model.response = {"usage": {"prompt_tokens": 9}}
    with pytest.raises(ValueError, match="length mismatch"):
        asyncio.run(infer(body, output))


def test_backend_accepts_existing_ids_and_requires_tokenizer_path(backend):
    with pytest.raises(ValueError, match="tokenizer path"):
        backend.FixedInputVLLMAPI(fixed_input_tokens=8)
    model = backend.FixedInputVLLMAPI(path="/model", fixed_input_tokens=2)
    output = SimpleNamespace(extra_details_data={})
    assert asyncio.run(model.get_request_body([3, 4, 5], 3, output))["prompt"] == [4, 5]
    with pytest.raises(TypeError, match="string or a list"):
        asyncio.run(model.get_request_body({"messages": []}, 3, output))


def test_failed_length_check_cannot_pass_a_benchmark_without_a_baseline(tmp_path, backend):
    model = backend.FixedInputVLLMAPI(path="/model", fixed_input_tokens=8, fixed_input_artifact_dir=str(tmp_path))
    output = SimpleNamespace(extra_details_data={}, success=False)
    body = asyncio.run(model.get_request_body("question", 3, output))
    with pytest.raises(RuntimeError, match="all requested"):
        fixed.validate_fixed_input_artifacts(str(tmp_path), 1)
    model.response = {"usage": {"prompt_tokens": 9}}
    with pytest.raises(ValueError, match="length mismatch"):
        asyncio.run(model.stream_infer(body, output))
    with pytest.raises(RuntimeError, match="failed or unvalidated"):
        fixed.validate_fixed_input_artifacts(str(tmp_path), 1)
    model.response = {"usage": {"prompt_tokens": 8}}
    asyncio.run(model.stream_infer(body, output))
    fixed.validate_fixed_input_artifacts(str(tmp_path), 1)
    assert len(list(tmp_path.glob("*.json"))) == 1
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["final_input_token_ids"] == body["prompt"]


def load_runner_method(name, namespace):
    tree = ast.parse((ROOT / "tools/aisbench.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AisbenchRunner")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "tools/aisbench.py", "exec"), namespace)
    return namespace[name]


def test_real_runner_generates_importable_fixed_backend_config(tmp_path, backend):
    (tmp_path / "vllm_api_general_stream.py").write_text("models = [dict(\n    model='',\n    path='',\n)]\n")
    attrs = dict.fromkeys(
        [
            "top_p",
            "top_k",
            "seed",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "thinking",
            "reasoning_effort",
            "temperature",
            "no_pred",
        ],
        None,
    )
    runner = SimpleNamespace(
        **attrs,
        model="/model",
        port=8100,
        host_ip="localhost",
        max_out_len=1024,
        batch_size=60,
        trust_remote_code=True,
        request_conf="vllm_api_general_stream",
        task_type="performance",
        model_path="/model",
        request_rate=0,
        fixed_input_tokens=2048,
        tokenizer_mode="kimi_k3",
    )
    method = load_runner_method(
        "_init_request_conf",
        dict(os=os, re=re, REQUEST_CONF_DIR=str(tmp_path), append_fixed_input_config=fixed.append_fixed_input_config),
    )
    method(runner)
    generated = (tmp_path / "vllm_api_general_stream_custom.py").read_text()
    scope = {}
    exec(compile(generated, "generated.py", "exec"), scope)
    config = scope["models"][0]
    assert config["type"] is backend.FixedInputVLLMAPI
    assert config["fixed_input_tokens"] == 2048
    assert config["tokenizer_mode"] == "kimi_k3"
    assert config["path"] == "/model"
    assert fixed.append_fixed_input_config("legacy", None, "auto") == "legacy"


def test_aisbench_child_can_import_repo_backend(tmp_path):
    calls = []
    method = load_runner_method(
        "_run_aisbench_task",
        dict(
            os=os,
            Path=Path,
            shlex=shlex,
            __file__=str(ROOT / "tools/aisbench.py"),
            subprocess=SimpleNamespace(Popen=lambda *args, **kwargs: calls.append(kwargs)),
        ),
    )
    runner = SimpleNamespace(
        dataset_conf="gsm8k/perf",
        task_type="performance",
        request_conf="vllm_api_general_stream",
        num_prompts=100,
        benchmark_work_dir=str(tmp_path),
        profile_marker=None,
        fixed_input_tokens=2048,
    )
    method(runner)
    assert str(ROOT) in calls[0]["env"]["PYTHONPATH"].split(os.pathsep)


def test_kimi_yaml_has_one_spec_per_role_and_one_fixed_benchmark():
    files = list((ROOT / "tests").glob("*/cases/models/configs/Kimi/Kimi-K3-W4A8-E256-A5-PD-Perf.yaml"))
    assert len(files) == 1
    config = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert list(config["benchmarks"]) == ["perf_gsm8k_first100_c60"]
    case = config["benchmarks"]["perf_gsm8k_first100_c60"]
    assert fixed.fixed_input_options(case) == (2048, "kimi_k3")
    assert [case[key] for key in ("num_prompts", "batch_size", "repeat_count")] == [100, 60, 5]
    for template in config["templates"]:
        args = template["server_cmd_template"]
        assert args.count("--speculative-config") == 1
        spec = json.loads(args[args.index("--speculative-config") + 1])
        assert spec["method"] == "ngram"
        assert spec["synthetic_acceptance_rates"] == [0.5]
