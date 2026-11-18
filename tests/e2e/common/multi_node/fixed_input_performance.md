# Fixed-input performance benchmarks

Set `fixed_input_tokens` on a performance benchmark to send exactly that many
input token IDs through the vLLM completions API. For example:

```yaml
benchmarks:
  perf_gsm8k_first100_c60:
    case_type: performance
    request_conf: vllm_api_general_stream
    fixed_input_tokens: 2048
    tokenizer_mode: kimi_k3
    num_prompts: 100
    batch_size: 60
    repeat_count: 5
    max_out_len: 1024
```

Keep the existing dataset, model paths, and other benchmark fields. This is an
opt-in AISBench backend: accuracy and chat configurations without
`fixed_input_tokens` retain their existing behavior. Fixed inputs require
`vllm_api_general` or `vllm_api_general_stream` and a local tokenizer path.
`tokenizer_mode` defaults to `auto`; custom modes such as `kimi_k3` use the
matching tokenizer from the installed vLLM version.

The backend tokenizes the final dataset prompt without special tokens, prepends
repeated padding token IDs to short prompts, and keeps the final
`fixed_input_tokens` IDs of long prompts. It sends the IDs directly with
`add_special_tokens=false`; it does not decode and re-encode them or add a chat
template. This changes the input distribution and is intended for performance
experiments, not accuracy evaluation.

AISBench still controls concurrency, retries, and performance reporting. The
final input token IDs are retained in the request's `extra_details_data` under
`fixed_input`. Successful responses must contain `usage.prompt_tokens` equal to
the configured length. Missing or mismatched usage makes the request fail. Per-request validation records
are also written to an isolated `fixed-input-*` directory in the benchmark work
directory (or the system temporary directory for legacy runs). The runner checks
these records before accepting a result, including when no throughput baseline
is configured. A successful retry overwrites that request's earlier failure.
`max_out_len` remains an output limit; fixed input alone does not guarantee a
fixed output length. Streaming event intervals are chunk gaps, not necessarily
individual-token ITL under speculative decoding.

Configure synthetic speculative acceptance on both server templates, rather than
appending it in a launch script:

```yaml
server_cmd_template:
  - --speculative-config
  - '{"method":"ngram","num_speculative_tokens":1,"prompt_lookup_min":1,"prompt_lookup_max":5,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.5]}'
```

This sets a synthetic acceptance target for performance testing and can change
output semantics. It does not guarantee that measured acceptance is exactly 50%.
Use a supported model runner: the tested 0.30.0 image needs Model Runner V1 for
ngram, while the tested main image supports Model Runner V2. The YAML must be the
single source of the speculative configuration; launch scripts must not append
another `--speculative-config`.
