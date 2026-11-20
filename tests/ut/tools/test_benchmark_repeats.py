# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real repeat/parser/runner code without NPU dependencies."""

import ast
import importlib.util
import json
import logging
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("benchmark_repeats", ROOT / "tools/benchmark_repeats.py")
assert SPEC is not None and SPEC.loader is not None
repeats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repeats)


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def parse_cases(raw):
    parser = load_function(
        "tests/e2e/common/multi_node/external_dp/external_dp_config.py",
        "_parse_benchmarks",
        {"Any": object, "expand_benchmark_repeats": repeats.expand_benchmark_repeats},
        "ExternalDPConfigLoader",
    )
    return parser(raw)


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, "5", None])
def test_invalid_repeat_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        parse_cases({"benchmarks": {"case": {"repeat_count": count}}})


def test_legacy_default_and_config_isolation():
    cases = parse_cases({"benchmarks": {"legacy": {"case_type": "accuracy"}}})
    assert len(cases) == 1
    assert "repeat_index" not in cases[0]
    raw = {"benchmarks": {"case": {"repeat_count": 2, "nested": []}}}
    expanded = parse_cases(raw)
    expanded[0]["nested"].append("changed")
    assert expanded[1]["nested"] == []
    assert raw["benchmarks"]["case"]["nested"] == []


def test_real_accuracy_tiers_expand_independently():
    counts = []
    for suffix, prompts, batch, rounds in [("First20", 20, 1, 1), ("First100", 100, 60, 5)]:
        paths = list(ROOT.glob(f"tests/**/Kimi-K3-W4A8-E256-A3-PD-Accu-{suffix}.yaml"))
        assert len(paths) == 1
        raw = yaml.safe_load(paths[0].read_text())
        assert raw["test_name"].endswith(f"-{suffix}")
        assert len(raw["benchmarks"]) == 1
        cases = parse_cases(raw)
        assert len(cases) == rounds
        assert [case["repeat_index"] for case in cases] == list(range(1, rounds + 1))
        assert all(case["num_prompts"] == prompts and case["batch_size"] == batch for case in cases)
        counts.append(sum(case["num_prompts"] for case in cases))
    assert counts == [20, 500]


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "..", "bad/name"])
def test_unsafe_artifact_paths_rejected(name):
    with pytest.raises(ValueError, match="safe directory"):
        repeats.expand_benchmark_repeats([{"benchmark_name": name, "repeat_count": 1}])


def test_artifact_collision_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        repeats.expand_benchmark_repeats(
            [
                {"benchmark_name": "case", "repeat_count": 1},
                {"benchmark_name": "case", "repeat_count": 2},
            ]
        )


def run_cases(cases, runner):
    namespace = {
        "AisbenchRunner": runner,
        "logging": logging,
        "prepare_repeat_artifacts": repeats.prepare_repeat_artifacts,
        "record_repeat_result": repeats.record_repeat_result,
        "write_repeat_summary": repeats.write_repeat_summary,
    }
    run = load_function("tools/aisbench.py", "run_aisbench_cases", namespace)
    return run("model", 1999, cases)


def test_failure_continues_with_distinct_outputs_and_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = parse_cases(
        {
            "benchmarks": {
                "small": {"case_type": "accuracy", "repeat_count": 1},
                "large": {"case_type": "accuracy", "repeat_count": 5},
            }
        }
    )
    visited = []

    class Runner:
        def __init__(self, **kwargs):
            case = kwargs["aisbench_config"]
            directory = Path(case["benchmark_work_dir"])
            assert not (directory / "outputs").exists()
            (directory / "outputs").mkdir()
            (directory / "outputs/predictions.json").write_text(json.dumps(case["repeat_index"]))
            visited.append(directory)
            if case["source_benchmark_name"] == "large" and case["repeat_index"] == 2:
                raise RuntimeError("intentional round failure")
            self.result = float(60 + case["repeat_index"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    with pytest.raises(AssertionError, match="some aisbench cases failed"):
        run_cases(cases, Runner)
    assert len(set(visited)) == 6
    assert len({d.parent.parent for d in visited}) == 1
    statuses = [json.loads((d / "status.json").read_text()) for d in visited]
    assert [s["outcome"] for s in statuses] == ["passed", "passed", "failed", "passed", "passed", "passed"]
    summary = json.loads((visited[0].parent.parent / "summary.json").read_text())
    large = next(v for v in summary.values() if len(v["rounds"]) == 5)
    assert large["failed"] == 1
    assert large["accuracy_mean"] == 63.25
    assert large["accuracy_min"] == 61
    assert large["accuracy_max"] == 65
    second = repeats.prepare_repeat_artifacts(cases)
    assert Path(second[0]["benchmark_work_dir"]) != visited[0]
    assert "benchmark_work_dir" not in cases[0]


def test_success_returns_six_results_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = parse_cases({"benchmarks": {"small": {"repeat_count": 1}, "large": {"repeat_count": 5}}})

    class Runner:
        def __init__(self, **kwargs):
            self.result = kwargs["aisbench_config"]["repeat_index"]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    assert run_cases(cases, Runner) == [1, 1, 2, 3, 4, 5]


def test_legacy_cases_keep_output_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = parse_cases({"benchmarks": {"legacy": {"case_type": "accuracy"}}})

    class Runner:
        def __init__(self, **kwargs):
            assert "benchmark_work_dir" not in kwargs["aisbench_config"]
            self.result = 60.0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    assert run_cases(cases, Runner) == [60.0]
    assert not (tmp_path / "outputs").exists()


def test_aisbench_command_uses_unique_work_dir(tmp_path):
    calls = []
    namespace = {
        "Path": Path,
        "shlex": shlex,
        "subprocess": SimpleNamespace(Popen=lambda command, **kwargs: calls.append(command)),
    }
    run = load_function("tools/aisbench.py", "_run_aisbench_task", namespace, "AisbenchRunner")
    runner = SimpleNamespace(
        dataset_conf="gsm8k/test",
        task_type="accuracy",
        request_conf="chat",
        num_prompts=100,
        benchmark_work_dir=str(tmp_path / "with spaces"),
        profile_marker=None,
    )
    run(runner)
    command = shlex.split(calls[0])
    assert command[command.index("--work-dir") + 1] == str(tmp_path / "with spaces/outputs")
    assert command[command.index(">") + 1] == str(tmp_path / "with spaces/output_accuracy.txt")


def test_result_entry_keeps_round_identity():
    namespace = {"Any": object}
    task_passed = load_function("tests/e2e/common/multi_node/benchmark_results.py", "task_passed", namespace)
    namespace["task_passed"] = task_passed
    build = load_function("tests/e2e/common/multi_node/benchmark_results.py", "build_task_entry", namespace)
    entry = build(
        "case",
        {
            "case_type": "accuracy",
            "benchmark_name": "case-round-02-of-05",
            "repeat_index": 2,
            "repeat_count": 5,
        },
        63.0,
    )
    assert entry["repeat_index"] == 2
    assert entry["repeat_count"] == 5
    assert entry["benchmark_name"] == "case-round-02-of-05"
