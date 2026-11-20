# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial benchmark repetitions on one running service, with isolated artifacts."""

import json
import math
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any


def expand_benchmark_repeats(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand explicitly repeated cases; leave legacy cases unchanged."""
    expanded = []
    names = set()
    for case in cases:
        if "repeat_count" not in case:
            expanded.append(case)
            continue
        count = case["repeat_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("repeat_count must be a positive integer")
        name = str(case.get("benchmark_name", case.get("case_name", "unknown")))
        artifact = str(case.get("benchmark_artifact_name", name))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact) or artifact in {".", ".."}:
            raise ValueError("Repeated benchmark artifact name must be a safe directory name")
        if artifact in names:
            raise ValueError(f"Duplicate repeated benchmark artifact name: {artifact}")
        names.add(artifact)
        for index in range(1, count + 1):
            item = deepcopy(case)
            item["source_benchmark_name"] = name
            item["repeat_index"] = index
            item["repeat_artifact_group"] = artifact
            suffix = f"-round-{index:02d}-of-{count:02d}"
            item["benchmark_name"] = name + suffix
            item["case_name"] = name + suffix
            item["benchmark_artifact_name"] = artifact + suffix
            expanded.append(item)
    return expanded


def prepare_repeat_artifacts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every explicit round a fresh absolute AISBench work directory."""
    if not any("repeat_index" in case for case in cases if case):
        return cases
    parent = Path("outputs/benchmark_repeats").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    prepared = []
    for case in cases:
        if not case or "repeat_index" not in case:
            prepared.append(case)
            continue
        item = deepcopy(case)
        directory = run_dir / item["repeat_artifact_group"] / f"round-{item['repeat_index']:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        item["benchmark_work_dir"] = str(directory)
        (directory / "config.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        prepared.append(item)
    return prepared


def record_repeat_result(case: dict[str, Any], result: Any = None, error: Exception | None = None) -> None:
    if "benchmark_work_dir" not in case:
        return
    status = {
        "benchmark": case["source_benchmark_name"],
        "round": case["repeat_index"],
        "repeat_count": case["repeat_count"],
        "outcome": "failed" if error is not None else "passed",
    }
    if error is not None:
        status["error"] = f"{type(error).__name__}: {error}"
    elif case.get("case_type") == "accuracy":
        status["accuracy"] = float(result)
    path = Path(case["benchmark_work_dir"]) / "status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def write_repeat_summary(cases: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    run_dir = None
    for case in cases:
        if not case or "benchmark_work_dir" not in case:
            continue
        directory = Path(case["benchmark_work_dir"])
        run_dir = directory.parent.parent
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        groups.setdefault(case["repeat_artifact_group"], []).append(status)
    if run_dir is None:
        return
    summary = {}
    for name, rounds in groups.items():
        scores = [r["accuracy"] for r in rounds if "accuracy" in r and math.isfinite(r["accuracy"])]
        entry = {
            "rounds": rounds,
            "passed": sum(r["outcome"] == "passed" for r in rounds),
            "failed": sum(r["outcome"] == "failed" for r in rounds),
        }
        if scores:
            entry["successful_accuracy_rounds"] = len(scores)
            entry["accuracy_mean"] = mean(scores)
            entry["accuracy_min"] = min(scores)
            entry["accuracy_max"] = max(scores)
        summary[name] = entry
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
