# Kimi-K3.1 W4A8 A3 expert pruning

`prune_experts_a3.py` implements the prefix-expert procedure in the repository's
Chinese E32 pruning report. It writes an independent checkpoint and never starts
an inference server. Python 3.10 or later is required; generation and byte
verification use only the standard library.

## Usage

Run from the repository root on the current A3 server. The commands use its
host paths after moving the original weights under `weights/Eco-Tech`.
`--output` names a new model directory under `/data/c30087248/weights`, not
the existing weights root itself. The output directory must not already exist.

```bash
python tools/kimi_k31/prune_experts_a3.py \
  --source /data/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8 \
  --output /data/c30087248/weights/Kimi-K3.1-Pro-w4a8-E32 \
  --keep 32 --plan-only

python tools/kimi_k31/prune_experts_a3.py \
  --source /data/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8 \
  --output /data/c30087248/weights/Kimi-K3.1-Pro-w4a8-E32 --keep 32

python tools/kimi_k31/prune_experts_a3.py \
  --source /data/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8 \
  --output /data/c30087248/weights/Kimi-K3.1-Pro-w4a8-E32 --keep 32 --verify-only
```

Keep the source checkpoint unchanged throughout generation and verification.
Plan mode scans headers and metadata without creating output files. Generation
needs space for the retained tensor data, headers, auxiliary files and reports.
The free-space check covers tensor bytes; leave additional space for the rest.

## Supported layout

The source must contain `config.json`,
`quant_model_weights.safetensors.index.json`,
`quant_model_description.json`, and the indexed safetensors shards.
The script derives layer count, hidden size and original expert count from
`text_config`, and requires:

- One initial dense layer, MoE at every subsequent layer, two shared experts,
  and a single expert group.
- Separate numbered experts under
  `language_model.model.layers.<layer>.block_sparse_moe.experts.<expert>`.
- Exactly `w1`, `w2`, and `w3`, each with `weight`, `weight_scale`,
  `weight_offset`, and `scale_bias` for every original expert.
- BF16 MoE gate weights shaped `[original_experts, hidden_size]` and FP32
  `e_score_correction_bias` shaped `[original_experts]`.
- A flat quantization description whose tensor keys match the index, plus
  the global fields listed in the original report.
- Regular source files without symbolic links and safetensors dtypes listed
  in the script. Unsupported layouts fail before publishing a checkpoint.

For the documented model, the original values are 93 layers, hidden size 7168,
896 routed experts and top-k 16. `--keep 32` keeps expert IDs 0–31 and the
matching router rows. Shared experts, attention (including attention gates),
vision and other tensors retain their dtype, shape and bytes. Only
`text_config.num_experts` changes in the configuration. No quantization tags,
scales or offsets are recomputed. This does not convert an MXFP model into an
A3-compatible quantization format.

## Output and recovery

Generation acquires `<output>.lock`, writes into `<output>.partial`, and
publishes the directory only after comparing every retained output tensor to
its source byte range. Reads use buffers of at most 16 MiB. Metadata and the
full tensor index remain in RAM, so memory usage scales with tensor count.

Each nonempty source shard becomes one output shard; empty shards are omitted.
The index and quantization description are rebuilt. Auxiliary files are copied
and hashed. Reports include `pruning_manifest.json`,
`pruning_tensor_manifest.jsonl`, `SHA256SUMS.json`, and `validation_report.json`.
Source file size, modification time, inode and mode are checked before and after
generation; removed expert payloads are not read or fully hashed.

Existing final directories, partial directories and locks are rejected.
Failed runs retain partial output for inspection. There is no automatic resume.
After interruption, confirm the original process has stopped before manually
handling its lock or partial output. Use a new output path for a fresh run.

## Validation scope

Synthetic-file tests cover byte preservation, router slicing, metadata updates,
missing tensors, malformed headers, output protection and corruption detection.
Run them without loading the repository's NPU-dependent test initialization:

```bash
python -c 'import runpy; runpy.run_path("tests/ut/tools/test_prune_kimi_k31_a3.py", run_name="__main__")'
```

This newly generated implementation has not processed the full production
checkpoint. The historical report's successful run belongs to the original
server scripts. NPU model loading, CPU numerical routing, inference and accuracy
evaluation are separate validation steps; the script's byte-level report does
not mark them as passed.

### Validation and capacity checks

Generation and `--verify-only` compare retained tensor bytes and auxiliary files
(including tokenizer assets) against the source. Verification also requires
`SHA256SUMS.json`, checks its complete expected file set, and verifies every listed
file hash. The report separates `tensors_verified` (count) from
`tensor_bytes_verified` (bytes), and records `auxiliary_files_verified`.

The capacity estimate includes retained payloads, auxiliary files, metadata and a
conservative margin. Available space may still change during generation. Existing
pruning reports in the source are rejected before writing any output. Failed
writes leave `.partial` for inspection, release the lock on handled exceptions,
and do not publish the final directory. Abrupt termination can leave the lock;
there is no automatic resume or cleanup.
