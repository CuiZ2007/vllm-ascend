# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-length completions requests for performance benchmarks."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PADDING_TEXT = "Solve the following arithmetic problem carefully. "
COMPLETIONS_CONFIGS = ("vllm_api_general", "vllm_api_general_stream")


def positive_token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("fixed_input_tokens must be a positive integer")
    return value


def fixed_input_options(config: Mapping[str, Any]) -> tuple[int | None, str]:
    count = config.get("fixed_input_tokens")
    mode = config.get("tokenizer_mode", "auto")
    if "fixed_input_tokens" not in config:
        return None, mode
    count = positive_token_count(count)
    if config.get("case_type") != "performance":
        raise ValueError("fixed_input_tokens is supported only for performance benchmarks")
    if config.get("request_conf") not in COMPLETIONS_CONFIGS:
        raise ValueError(
            "fixed_input_tokens requires vllm_api_general or vllm_api_general_stream (token-ID completions)"
        )
    if not isinstance(mode, str) or not mode:
        raise ValueError("tokenizer_mode must be a nonempty string")
    return count, mode


def fit_prompt_tokens(token_ids: Sequence[int], padding_ids: Sequence[int], count: int) -> list[int]:
    """Prepend padding to short prompts; keep the suffix of long prompts.

    Token IDs are sent directly to /v1/completions, without a decode/re-encode
    round trip or a server-side chat template adding extra tokens.
    """
    positive_token_count(count)
    ids = list(token_ids)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError("Prompt must contain nonnegative integer token IDs")
    if len(ids) >= count:
        return ids[-count:]
    padding = list(padding_ids)
    if not padding or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in padding):
        raise ValueError("Padding must contain nonnegative integer token IDs")
    missing = count - len(ids)
    repeats = (missing + len(padding) - 1) // len(padding)
    return (padding * repeats)[:missing] + ids


def validate_prompt_usage(usage: Mapping[str, Any], expected: int) -> None:
    actual = usage.get("prompt_tokens")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise ValueError(f"Fixed input length mismatch: expected {expected} prompt tokens, server reported {actual!r}")


def validate_cached_usage(usage: Mapping[str, Any], minimum: int | None) -> int | None:
    """Reject decode-focused measurements without sufficient reported cache hits."""
    if minimum is None:
        return None
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
    if isinstance(cached, bool) or not isinstance(cached, int) or cached < minimum:
        raise ValueError(f"Prefix cache requirement failed: expected at least {minimum} cached tokens, got {cached!r}")
    return cached


def validate_fixed_input_artifacts(directory: str, expected_requests: int | None) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in Path(directory).glob("*.json")]
    # AISBench may issue additional warmup requests using the same backend.
    if len(records) < (expected_requests or 1):
        raise RuntimeError("Fixed-input benchmark did not validate all requested inputs")
    if any(not record.get("validated") for record in records):
        raise RuntimeError("Fixed-input benchmark contains failed or unvalidated requests")


def append_fixed_input_config(
    content: str,
    count: int | None,
    tokenizer_mode: str,
    artifact_dir: str | None = None,
    min_cached_tokens: int | None = None,
) -> str:
    if min_cached_tokens is not None:
        positive_token_count(min_cached_tokens)
        if count is None or min_cached_tokens > count:
            raise ValueError("min_cached_tokens requires fixed_input_tokens >= min_cached_tokens")
    if count is None:
        return content
    positive_token_count(count)
    return content + (
        "\nfrom tools.aisbench_fixed_input import FixedInputVLLMAPI\n"
        "for _fixed_input_model in models:\n"
        "    _fixed_input_model.update(\n"
        "        type=FixedInputVLLMAPI,\n"
        f"        fixed_input_tokens={count!r},\n"
        f"        tokenizer_mode={tokenizer_mode!r},\n"
        f"        fixed_input_artifact_dir={artifact_dir!r},\n"
        f"        min_cached_tokens={min_cached_tokens!r},\n"
        "    )\n"
    )
