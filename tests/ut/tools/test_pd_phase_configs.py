"""Validate phase workloads and their nightly references without NPU dependencies."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("phase", ["PureP", "PureD"])
def test_phase_workload_and_nightly_registration(phase):
    paths = list(ROOT.glob(f"tests/**/Kimi-K3-W4A8-E256-A5-PD-Perf-{phase}.yaml"))
    assert len(paths) == 1
    path = paths[0]
    config = yaml.safe_load(path.read_text())
    assert config["num_nodes"] == 2
    assert config["npu_per_node"] == 8
    cases = list(config["benchmarks"].values())
    assert len(cases) == 2
    warmup, measured = cases
    assert warmup["repeat_count"] == 1 and measured["repeat_count"] == 5
    assert measured["num_prompts"] == 100 and measured["batch_size"] == 60
    assert warmup["fixed_input_tokens"] == measured["fixed_input_tokens"] == 2048
    assert warmup["max_out_len"] == 1
    assert measured["max_out_len"] == (1 if phase == "PureP" else 1024)
    assert measured["threshold"] is None
    assert measured["input_throughput_threshold"] is None
    assert warmup["dataset_conf"] == measured["dataset_conf"]
    assert warmup["dataset_path"] == measured["dataset_path"]
    if phase == "PureD":
        assert warmup["num_prompts"] == measured["num_prompts"]
        assert measured["min_cached_tokens"] == 1536
        assert "min_cached_tokens" not in warmup
    for template in config["templates"]:
        args = template["server_cmd_template"]
        assert int(args[args.index("--max-model-len") + 1]) >= 2048 + measured["max_out_len"]
        if phase == "PureP":
            assert "--enable-prefix-caching" in args
            assert "--no-enable-prefix-caching" not in args
            assert "--speculative-config" not in args
        else:
            assert "--enable-prefix-caching" in args
            assert "--enable-prompt-tokens-details" in args
            spec = json.loads(args[args.index("--speculative-config") + 1])
            assert spec["synthetic_acceptance_rates"] == [0.5]
    nightly = yaml.safe_load((ROOT / ".github/workflows/configs/nightly_config.yaml").read_text())
    entries = nightly["a5"]["multi_node"]["test_config"]
    matches = [e for e in entries if e.get("config_file_path") == path.name]
    assert len(matches) == 1
    assert ROOT / matches[0]["config_base_path"] / path.name == path
    assert matches[0]["size"] == 2
