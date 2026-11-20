# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-concurrency, dataset-backed functional checks for OpenAI chat endpoints."""

import json
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_questions(path: str, count: int) -> list[str]:
    positive_int(count, "num_prompts")
    source = Path(path)
    if source.is_dir():
        source = source / "test.jsonl"
    questions = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Dataset row must contain a nonempty question")
            questions.append(question)
            if len(questions) == count:
                break
    if len(questions) != count:
        raise ValueError(f"Dataset has only {len(questions)} rows; requested {count}")
    return questions


def validate_chat_response(response: Any, max_tokens: int) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("object") != "chat.completion":
        raise ValueError("Expected a chat.completion object")
    if not isinstance(response.get("id"), str) or not response["id"]:
        raise ValueError("Missing response id")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Expected exactly one response choice")
    choice = choices[0]
    if choice.get("index") != 0 or choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Invalid choice index or finish reason")
    message = choice.get("message") or {}
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected an assistant message")
    texts = [message.get(key) for key in ("content", "reasoning_content", "reasoning")]
    if not any(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Empty assistant response, including reasoning")
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("Missing token usage")
    prompt = positive_int(usage.get("prompt_tokens"), "prompt_tokens")
    completion = positive_int(usage.get("completion_tokens"), "completion_tokens")
    total = positive_int(usage.get("total_tokens"), "total_tokens")
    if completion > max_tokens or total != prompt + completion:
        raise ValueError("Inconsistent or excessive token usage")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "finish_reason": choice["finish_reason"],
        "has_final_content": isinstance(message.get("content"), str) and bool(message["content"].strip()),
    }


def request_one(url: str, model: str, index: int, question: str, options: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    record = {"index": index, "question": question, "passed": False}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "max_tokens": options["max_out_len"],
        "temperature": options.get("temperature", 0),
        "top_p": options.get("top_p", 1.0),
        "seed": options.get("seed", 42),
    }
    try:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        # Local inference traffic must not inherit an Internet proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=options.get("request_timeout", 3000)) as response:
            record["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            data = json.load(response)
        record["response"] = data
        record.update(validate_chat_response(data, options["max_out_len"]))
        record["passed"] = True
    except urllib.error.HTTPError as error:
        record.update(http_status=error.code, error=f"HTTPError: {error.code}")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["elapsed_seconds"] = time.monotonic() - started
    return record


class FunctionalRunner:
    """Run each selected question once, retaining every failure and raw response."""

    def __init__(self, model: str, port: int, aisbench_config: dict[str, Any], host_ip: str = "localhost"):
        options = dict(aisbench_config)
        count = positive_int(options.get("num_prompts"), "num_prompts")
        concurrency = positive_int(options.get("batch_size"), "batch_size")
        positive_int(options.get("max_out_len"), "max_out_len")
        positive_int(options.get("request_timeout", 3000), "request_timeout")
        questions = load_questions(options.get("dataset_path_local") or options["dataset_path"], count)
        directory = options.get("benchmark_work_dir")
        if directory:
            work_dir = Path(directory)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            parent = Path("outputs/functional")
            parent.mkdir(parents=True, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
        host = f"[{host_ip}]" if ":" in host_ip and not host_ip.startswith("[") else host_ip
        url = f"http://{host}:{port}/v1/chat/completions"
        started = time.monotonic()

        def run(item):
            index, question = item
            record = request_one(url, model, index, question, options)
            path = work_dir / f"request-{index:04d}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            return record

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            records = list(executor.map(run, enumerate(questions, 1)))
        succeeded = sum(record["passed"] for record in records)
        self.result = {
            "requests": count,
            "succeeded": succeeded,
            "failed": count - succeeded,
            "concurrency": concurrency,
            "length_finished": sum(record.get("finish_reason") == "length" for record in records),
            "with_final_content": sum(record.get("has_final_content", False) for record in records),
            "elapsed_seconds": time.monotonic() - started,
            "artifact_dir": str(work_dir),
        }
        (work_dir / "functional-summary.json").write_text(json.dumps(self.result, indent=2), encoding="utf-8")
        if succeeded != count:
            raise AssertionError(f"Functional requests failed: {count - succeeded}/{count}; artifacts: {work_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
