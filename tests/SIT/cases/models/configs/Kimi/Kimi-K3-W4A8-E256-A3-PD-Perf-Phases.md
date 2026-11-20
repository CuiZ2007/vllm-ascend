# Kimi K3 E256 PD phase performance workloads

These draft workloads run one prefiller and one decoder. They exercise the full
PD request path; they do not isolate a node from KV transfer or client overhead.

| Workload | Fixed input | Output | Prefix caching | Measured requests | Concurrency | Rounds |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| PureP | 2048 | 1 | Enabled | 100 | 60 | 5 |
| PureD | 2048 | 1024 | Enabled on both roles | 100 | 60 | 5 |

PureP has a four-request, one-token warmup. It disables speculative decoding.
Prefix caching is enabled on both roles. Shared padding and repeated prompts may
reuse KV, including across measured rounds. This is a cache-enabled short-output
workload, not a cold-prefill compute baseline. Report actual cache hits alongside
TTFT and input throughput; input tokens can include cached work.
Use input throughput and TTFT; TTFT still includes queueing, KV transfer and the
first decode token. Do not interpret one-token TPOT as a meaningful decode metric.

PureD first warms the exact same first 100 GSM8K prompts, at concurrency 60, with
one output token. The fixed-input tokenizer and dataset configuration are identical
between warmup and measurement. With 768-token blocks, the last partial block is
not a reusable full block. Each measured response must report at least 1536 cached
input tokens (75% of 2048); missing or insufficient cache usage fails validation.
Check the actual cached-token records, eviction and P/D cache behavior on hardware.
If this image does not expose usage.prompt_tokens_details.cached_tokens, this
workload requires compatible server reporting before its results can be accepted.
Both roles enable --enable-prompt-tokens-details to request cache usage reporting.
In PD deployments the reported count may include externally transferred KV, not
just local prefix reuse. Cross-check P-side prefix-cache counters/logs and KV
transfer metrics; this response check alone does not prove low Prefill compute.
No cache hit or performance claim is made before hardware validation.

PureD retains the existing one-token ngram speculative setup with synthetic
acceptance 0.5. Main uses V2 and 0.30.0 uses V1 as in the source Task4 performance
case. These synthetic outputs are for performance only, not accuracy.

Use output throughput and TPOT for PureD. TTFT remains an end-to-end measurement.
Inter-chunk arrival gaps are not per-token ITL when one speculative chunk contains
multiple tokens. Warmup and measured rounds have distinct artifact groups; report
only the five measured rounds. Keep failures visible and never average only the
successful subset without reporting excluded rounds.

Performance baselines and gates are unset until hardware measurements establish
references. Archive AISBench reports, fixed-input/cache validation JSON, server
logs and the exact image digest. Verify actual output lengths in the reports;
performance requests retain ignore_eos=True from the runner.

Keep the existing mixed PD-Perf case as the end-to-end reference. All three cases now use 2048-token input contexts, but their output lengths,
cache policies and speculative settings differ; do not attribute metric differences
to the model version alone.
All performance cases, including the original mixed PD-Perf case, now enable
prefix caching. Their cache state and warmup policies still differ.
Use identical datasets, concurrency, phase workload and cache policy when comparing
A3/A5 or main/0.30.0; retain the documented Runner and speculative differences.

The model and dataset use the YAML container paths. Mount the corresponding shared
weights and GSM8K directories there before running. Both new YAML files are
registered as two-node nightly cases alongside the existing PD performance case.
