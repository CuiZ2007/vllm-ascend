# Kimi K3 E256 functional requests

The single-node mixed and two-node 1P1D YAML files each execute exactly one round
of the first 500 nonempty test.jsonl records, with at most 60 requests in flight.
Only the original question field is sent as one user message; there are no
few-shot examples, artificial padding, gold answers, or synthetic acceptance.
The model can generate up to 1024 tokens at temperature 0, top_p 1, seed 42.
The service uses the existing low-thinking chat template and accuracy-mode setup.
Prefix caching remains disabled as in the accuracy service templates.

Each selected question is sent once, without application-level retries. All 500
requests must return HTTP 200, a chat.completion object with a nonempty id, one
assistant choice with index 0, stop/length finish reason, and nonempty content or
reasoning. Prompt and completion token counts must be positive integers, completion
must not exceed 1024, and total usage must equal prompt plus completion usage.
Length-finished responses are valid protocol results, but are counted separately.
Reasoning-only responses are also counted separately via with_final_content.
This does not test mathematical accuracy or require every answer to finish within
the output budget. It does not cover streaming, tool calling, or error-input APIs.

The shared functional runner bypasses AISBench evaluation and writes one JSON file
per request plus functional-summary.json under the benchmark round work directory.
It retains failed responses and still attempts every selected question. Any failed
request fails the case; successful subsets cannot silently pass. Aggregate nightly
results include request/success/failure counts. The request timeout is 300 seconds
per socket operation, and concurrency is an upper bound rather than a guaranteed
constant number of simultaneous requests.

Mount the existing weights and GSM8K dataset at the container paths in the YAML.
A3 uses TP16, A5 uses TP8. PD deploys node 0 as P and node 1 as D. The YAML files
are registered in each branch's nightly matrix. Real NPU execution is pending.
