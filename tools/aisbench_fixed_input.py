# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AISBench backend preserving its scheduler and metrics for fixed inputs."""

import json
from pathlib import Path
from uuid import uuid4

from ais_bench.benchmark.models import VLLMCustomAPI

from tools.fixed_input import (
    PADDING_TEXT,
    fit_prompt_tokens,
    positive_token_count,
    validate_cached_usage,
    validate_prompt_usage,
)


def load_fixed_input_tokenizer(path: str, mode: str, trust_remote_code: bool):
    # vLLM moved this helper between releases. Its custom modes are necessary
    # for tokenizers such as kimi_k3; AutoTokenizer is sufficient for auto.
    if mode == "auto":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        from vllm.transformers_utils.tokenizer import get_tokenizer

    return get_tokenizer(path, tokenizer_mode=mode, trust_remote_code=trust_remote_code)


class FixedInputVLLMAPI(VLLMCustomAPI):
    """Send exact token-ID prompts and reject missing/mismatched usage.

    Only performance configs select this backend. Accuracy/chat backends keep
    their existing behavior. AISBench still owns concurrency, retries and
    performance reporting; streaming chunk gaps remain different from ITL.
    """

    def __init__(
        self,
        *args,
        fixed_input_tokens: int,
        tokenizer_mode: str = "auto",
        fixed_input_artifact_dir: str | None = None,
        min_cached_tokens: int | None = None,
        **kwargs,
    ):
        self.fixed_input_tokens = positive_token_count(fixed_input_tokens)
        self.min_cached_tokens = min_cached_tokens
        if min_cached_tokens is not None:
            positive_token_count(min_cached_tokens)
            if min_cached_tokens > self.fixed_input_tokens:
                raise ValueError("min_cached_tokens cannot exceed fixed_input_tokens")
        self.fixed_input_artifact_dir = Path(fixed_input_artifact_dir) if fixed_input_artifact_dir else None
        if self.fixed_input_artifact_dir is not None:
            self.fixed_input_artifact_dir.mkdir(parents=True, exist_ok=True)
        path = kwargs.get("path", args[0] if args else "")
        if not path:
            raise ValueError("Fixed-input benchmarks require a local tokenizer path")
        self.fixed_input_tokenizer = load_fixed_input_tokenizer(
            path, tokenizer_mode, kwargs.get("trust_remote_code", False)
        )
        self.fixed_padding_ids = self.fixed_input_tokenizer.encode(PADDING_TEXT, add_special_tokens=False)
        super().__init__(*args, **kwargs)

    async def get_request_body(self, input_data, max_out_len, output, **args):
        body = await super().get_request_body(input_data, max_out_len, output, **args)
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt = self.fixed_input_tokenizer.encode(prompt, add_special_tokens=False)
        elif not isinstance(prompt, list):
            raise TypeError("Fixed-input completions require a string or a list of token IDs")
        ids = fit_prompt_tokens(prompt, self.fixed_padding_ids, self.fixed_input_tokens)
        body["prompt"] = ids
        body["add_special_tokens"] = False
        if body.get("stream"):
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        output.extra_details_data["fixed_input"] = {
            "validation_id": uuid4().hex,
            "expected_prompt_tokens": self.fixed_input_tokens,
            "final_input_token_ids": ids,
            "usage_validated": False,
            "min_cached_tokens": self.min_cached_tokens,
            "cached_tokens": None,
        }
        return body

    async def _parse_usage(self, data, output):
        await super()._parse_usage(data, output)
        usage = data.get("usage")
        if usage is not None:
            validate_prompt_usage(usage, self.fixed_input_tokens)
            cached = validate_cached_usage(usage, self.min_cached_tokens)
            output.extra_details_data["fixed_input"]["cached_tokens"] = cached
            output.extra_details_data["fixed_input"]["usage_validated"] = True

    @staticmethod
    def _require_usage(output):
        if output.success and not output.extra_details_data["fixed_input"]["usage_validated"]:
            output.success = False
            raise ValueError("Fixed-input benchmark response is missing usage.prompt_tokens")

    def _save_validation(self, output):
        if self.fixed_input_artifact_dir is None:
            return
        details = output.extra_details_data["fixed_input"]
        record = {
            **details,
            "validated": bool(output.success and details["usage_validated"]),
            "actual_prompt_tokens": getattr(output, "input_tokens", None),
        }
        path = self.fixed_input_artifact_dir / (details["validation_id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(path)

    async def stream_infer(self, request_body, output):
        output.extra_details_data["fixed_input"]["usage_validated"] = False
        try:
            await super().stream_infer(request_body, output)
            self._require_usage(output)
        except BaseException:
            output.success = False
            raise
        finally:
            self._save_validation(output)

    async def text_infer(self, request_body, output):
        output.extra_details_data["fixed_input"]["usage_validated"] = False
        try:
            await super().text_infer(request_body, output)
            self._require_usage(output)
        except BaseException:
            output.success = False
            raise
        finally:
            self._save_validation(output)
