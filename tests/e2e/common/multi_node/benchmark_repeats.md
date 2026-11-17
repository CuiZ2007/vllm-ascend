# Repeating benchmarks on an external DP service

Each entry under `benchmarks` can set `repeat_count` to a positive integer.
If omitted, the benchmark runs once with the existing output behavior.
An explicit count, including `1`, enables separate round artifacts.

```yaml
benchmarks:
  gsm8k_first20_c1:
    repeat_count: 1
    # Keep the dataset, generation, and verification settings here.
    num_prompts: 20
    batch_size: 1
  gsm8k_first100_c60:
    repeat_count: 5
    num_prompts: 100
    batch_size: 60
```

This example shows only the repeat and workload fields. Each benchmark still
needs its normal AISBench configuration. The loader expands the entries into
six ordered runs: the 20-question case once, then the 100-question case five
times, for 520 answers in total. The P/D engines and proxy are started once;
repeating a benchmark does not restart the service or clear its internal state.

Every explicit round gets a fresh absolute AISBench `--work-dir`. Outputs are
stored under:

```text
outputs/benchmark_repeats/run-<unique-id>/<test-name>__<benchmark>/round-01/
  config.json
  status.json
  output_accuracy.txt
  outputs/...
```

Predictions, generated AISBench configs, and summaries remain inside that
round's `outputs` directory. Each invocation uses a new run directory, including
when a test is repeated in the same second. Benchmark result entries include
the round name, index, and total count.

`summary.json` in the run directory records all round statuses and the mean,
minimum, and maximum accuracy of successful finite results. Failed rounds are
counted explicitly and excluded from accuracy statistics. Ordinary benchmark
exceptions are recorded and later rounds continue; any failure still fails the
overall test. Process interruption is not treated as successful completion.

This mechanism is part of external DP test orchestration. It is separate from
single-node `test_rounds`, which repeats the entire test and its service startup.
Do not run two benchmark invocations concurrently against the same AISBench
installation: generated request configuration files are still shared there.
