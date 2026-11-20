# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for dataset-backed concurrent functional requests and YAML wiring."""

import ast
import importlib.util
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("functional_requests", ROOT / "tools/functional_requests.py")
functional = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(functional)


def response(text="answer", completion=2):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": completion, "total_tokens": 10 + completion},
    }


@pytest.mark.parametrize("value", [0, -1, True, "60", None])
def test_invalid_positive_options(value):
    with pytest.raises(ValueError):
        functional.positive_int(value, "batch_size")


def test_short_or_invalid_dataset_cannot_silently_pass(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({"question": "q"}) + "\n")
    with pytest.raises(ValueError, match="only 1"):
        functional.load_questions(str(tmp_path), 2)
    p.write_text(json.dumps({"question": ""}) + "\n")
    with pytest.raises(ValueError, match="nonempty"):
        functional.load_questions(str(p), 1)


@pytest.mark.parametrize("kind", ["empty", "role", "usage", "excess", "finish", "object", "choices"])
def test_response_validation_rejects_invalid_outputs(kind):
    data = response()
    if kind == "empty":
        data["choices"][0]["message"]["content"] = ""
    if kind == "role":
        data["choices"][0]["message"]["role"] = "user"
    if kind == "usage":
        data["usage"]["prompt_tokens"] = True
    if kind == "excess":
        data["usage"]["completion_tokens"] = 100
    if kind == "finish":
        data["choices"][0]["finish_reason"] = None
    if kind == "object":
        data["object"] = "error"
    if kind == "choices":
        data["choices"] = []
    with pytest.raises(ValueError):
        functional.validate_chat_response(data, 8)


def test_reasoning_only_and_length_finish_are_reported_explicitly():
    data = response("")
    data["choices"][0]["message"]["reasoning_content"] = "thinking"
    data["choices"][0]["finish_reason"] = "length"
    result = functional.validate_chat_response(data, 8)
    assert result["finish_reason"] == "length" and not result["has_final_content"]


def test_exactly_first500_questions_once_without_gold_answers(tmp_path, monkeypatch):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("".join(json.dumps({"question": f"q{i}", "answer": "SECRET_GOLD"}) + "\n" for i in range(501)))
    seen = []
    lock = threading.Lock()

    def fake(url, model, index, question, options):
        with lock:
            seen.append((index, question))
        return {
            "index": index,
            "question": question,
            "passed": True,
            "finish_reason": "stop",
            "has_final_content": True,
        }

    monkeypatch.setattr(functional, "request_one", fake)
    runner = functional.FunctionalRunner(
        "model",
        8080,
        dict(
            dataset_path_local=str(dataset),
            num_prompts=500,
            batch_size=60,
            max_out_len=1024,
            benchmark_work_dir=str(tmp_path / "artifacts"),
        ),
    )
    assert sorted(seen) == [(i + 1, f"q{i}") for i in range(500)]
    assert runner.result["requests"] == runner.result["succeeded"] == 500
    assert runner.result["failed"] == 0
    assert len(list((tmp_path / "artifacts").glob("request-*.json"))) == 500


@pytest.mark.parametrize("failure", [False, True])
def test_real_http_concurrency_failure_retention_and_no_retries(tmp_path, failure):
    active = 0
    peak = 0
    seen = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal active, peak
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["stream"] is False
            assert len(body["messages"]) == 1
            question = body["messages"][0]["content"]
            with lock:
                active += 1
                peak = max(peak, active)
                seen.append(question)
            time.sleep(0.01)
            fail = failure and question == "q2"
            payload = json.dumps({"error": "intentional"} if fail else response()).encode()
            self.send_response(503 if fail else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            with lock:
                active -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("".join(json.dumps({"question": f"q{i}"}) + "\n" for i in range(12)))
    config = dict(
        dataset_path_local=str(dataset),
        num_prompts=12,
        batch_size=3,
        max_out_len=8,
        benchmark_work_dir=str(tmp_path / "artifacts"),
        request_timeout=10,
    )
    try:
        if failure:
            with pytest.raises(AssertionError, match="1/12"):
                functional.FunctionalRunner("model", server.server_port, config, "127.0.0.1")
        else:
            runner = functional.FunctionalRunner("model", server.server_port, config, "127.0.0.1")
            assert runner.result["failed"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert sorted(seen) == sorted(f"q{i}" for i in range(12))
    assert 1 < peak <= 3
    summary = json.loads((tmp_path / "artifacts/functional-summary.json").read_text())
    assert summary["failed"] == int(failure)
    assert len(list((tmp_path / "artifacts").glob("request-*.json"))) == 12


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def test_shared_runner_dispatches_functional_cases():
    class Runner:
        def __init__(self, **kwargs):
            self.result = {"succeeded": 500}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    namespace = dict(
        FunctionalRunner=Runner,
        logging=logging,
        prepare_repeat_artifacts=lambda cases: cases,
        record_repeat_result=lambda *a, **k: None,
        write_repeat_summary=lambda cases: None,
    )
    run = load_function("tools/aisbench.py", "run_aisbench_cases", namespace)
    assert run("model", 8080, [{"case_type": "functional"}]) == [{"succeeded": 500}]


@pytest.mark.parametrize(
    "path,name",
    [
        ("tests/e2e/common/single_node/test_single_node.py", "_task_passed"),
        ("tests/e2e/common/multi_node/benchmark_results.py", "task_passed"),
    ],
)
def test_summary_rejects_partial_functional_success(path, name):
    function = load_function(path, name, {"Any": object})
    config = dict(case_type="functional", num_prompts=500)
    assert function(config, dict(requests=500, succeeded=500, failed=0))
    assert not function(config, dict(requests=500, succeeded=499, failed=1))
    assert not function(config, "")


@pytest.mark.parametrize("pd", [False, True])
def test_functional_yaml_is_first500_concurrency60_one_round_and_registered(pd):
    stem = "Kimi-K3-W4A8-E256-A5" + ("-PD" if pd else "") + "-Functional-First500"
    paths = list(ROOT.glob(f"tests/**/{stem}.yaml"))
    assert len(paths) == 1
    raw = yaml.safe_load(paths[0].read_text())
    config = raw if pd else raw["test_cases"][0]
    assert len(config["benchmarks"]) == 1
    case = next(iter(config["benchmarks"].values()))
    assert case["case_type"] == "functional"
    assert (case["num_prompts"], case["batch_size"], case["repeat_count"]) == (500, 60, 1)
    assert "baseline" not in case and "fixed_input_tokens" not in case
    if pd:
        assert config["num_nodes"] == 2 and config["npu_per_node"] == 8
        args = [t["server_cmd_template"] for t in config["templates"]]
    else:
        assert config["test_rounds"] == 1 and config["test_content"] == []
        args = [config["server_cmd"]]
        assert int(args[0][args[0].index("--tensor-parallel-size") + 1]) == 8
    assert all("--speculative-config" not in a and "--no-enable-prefix-caching" in a for a in args)
    nightly = yaml.safe_load((ROOT / ".github/workflows/configs/nightly_config.yaml").read_text())
    entries = [
        e
        for groups in nightly.values()
        if isinstance(groups, dict)
        for group in groups.values()
        if isinstance(group, dict)
        for e in group.get("test_config", [])
    ]
    matches = [e for e in entries if e["name"] == stem]
    assert len(matches) == 1
    assert ROOT / matches[0]["config_base_path"] / matches[0]["config_file_path"] == paths[0]
