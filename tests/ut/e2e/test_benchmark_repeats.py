import importlib.util
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "tests/e2e/common/single_node/benchmark_repeats.py"
SPEC = importlib.util.spec_from_file_location("benchmark_repeats", HELPER)
assert SPEC is not None and SPEC.loader is not None
repeats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repeats)


def test_two_tiers_have_six_independent_results():
    cases = {
        "small": {"num_prompts": 20, "batch_size": 1, "repeat_count": 1},
        "large": {"num_prompts": 100, "batch_size": 60, "repeat_count": 5},
    }
    expanded = repeats.expand_benchmark_repeats(cases)
    assert list(expanded) == ["small", *[f"large_round{i:02d}" for i in range(1, 6)]]
    assert expanded["small"]["num_prompts"] == 20
    for i in range(1, 6):
        case = expanded[f"large_round{i:02d}"]
        assert (case["num_prompts"], case["batch_size"]) == (100, 60)
        assert case["case_name"] == f"large_round{i:02d}"
        assert "repeat_count" not in case
    expanded["large_round01"]["num_prompts"] = 1
    assert expanded["large_round02"]["num_prompts"] == 100
    assert cases["large"]["num_prompts"] == 100


@pytest.mark.parametrize("count", [0, -1, True, "5", 1.5])
def test_invalid_repeat_counts_are_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        repeats.expand_benchmark_repeats({"case": {"repeat_count": count}})


def test_default_and_disabled_cases_are_preserved():
    cases = {"enabled": {"batch_size": 1}, "disabled": None}
    assert repeats.expand_benchmark_repeats(cases) == cases
    assert repeats.expand_benchmark_repeats(None) is None


def test_generated_names_cannot_overwrite_configured_cases():
    with pytest.raises(ValueError, match="collides"):
        repeats.expand_benchmark_repeats({"case": {"repeat_count": 2}, "case_round01": {"batch_size": 1}})
