# Kimi-K3.1 W4A8 MXFP E32 A5 expert pruning

`prune_experts_a5.py` retains a prefix of the routed experts and the matching
router rows. It copies the MXFP payloads without dequantization or requantization.
Python 3.10 or later and the standard library are sufficient. Generation runs on
the host CPU and storage; it does not start an NPU inference server.

## Usage

Run from the repository root. These paths refer to the shared disk as mounted on
the inspected server. Confirm the mount on the execution host before running.
The output must be a new model directory, with no existing partial directory or
lock.

```bash
python tools/kimi_k31/prune_experts_a5.py \
  --source /mnt/share/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp \
  --output /mnt/share/c30087248/weights/Kimi-K3.1-Pro-w4a8-mxfp-E32 \
  --keep 32 --plan-only

python tools/kimi_k31/prune_experts_a5.py \
  --source /mnt/share/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp \
  --output /mnt/share/c30087248/weights/Kimi-K3.1-Pro-w4a8-mxfp-E32 \
  --keep 32

python tools/kimi_k31/prune_experts_a5.py \
  --source /mnt/share/c30087248/weights/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp \
  --output /mnt/share/c30087248/weights/Kimi-K3.1-Pro-w4a8-mxfp-E32 \
  --keep 32 --verify-only
```

Keep the source unchanged throughout planning, generation and verification.
Plan mode reads headers and metadata without writing a checkpoint. Generation
streams data in blocks of at most 16 MiB; its in-memory plan and tensor metadata
scale with tensor count. Reserve space for payloads, auxiliary files and reports,
and account for other writers to the shared filesystem.

## Supported checkpoint layout

The source must contain `config.json`,
`quant_model_weights.safetensors.index.json`, `quant_model_description.json`, and
exactly the top-level safetensors shards named in the index. Symbolic links,
special files and paths outside the source are rejected. Shard headers, lengths,
offsets, dtypes, shapes and index membership are checked before payload copying.

The supported text model has one initial dense layer, MoE at every subsequent
layer, two shared experts and a single expert group. The retained expert count
must be at least the configured top-k and below the original expert count.

Each routed expert has exactly six tensors: `w1`, `w2`, `w3`, each with `weight`
and `weight_scale`. All six must be `U8` with `W4A8_MXFP` quantization labels.
`group_size` must be 32. Shapes are derived from `moe_intermediate_size` and
`routed_expert_hidden_size`; both dimensions must be divisible by 32:

| Projection | Weight shape | Scale shape |
| --- | --- | --- |
| `w1`, `w3` | `[intermediate, routed_hidden / 2]` | `[intermediate, routed_hidden / 32]` |
| `w2` | `[routed_hidden, intermediate / 2]` | `[routed_hidden, intermediate / 32]` |

The MoE gate must be BF16 `[original_experts, hidden_size]` and its correction
bias F32 `[original_experts]`. Both retain their first `keep` rows or elements.
Attention gates and shared experts are preserved. Non-expert FP8 payloads,
including `F8_E4M3`, keep their original bytes and quantization labels.

## Output and verification

The script acquires `<output>.lock`, writes independent files to
`<output>.partial`, rebuilds the index and quantization description, and changes
only `text_config.num_experts`. Each nonempty source shard produces one output
shard; empty shards are omitted and the count is derived from the plan.

Auxiliary files are copied and hashed. `.cache`, `.git`, `.locks`, `__pycache__`,
files ending in `.tmp`, `.part`, `.lock`, and nested unindexed safetensors files
are excluded. Review the auxiliary contents before generating. Sources containing
top-level pruning reports are rejected to avoid pruning an existing output by
mistake.

Generation independently rereads all output tensors and compares SHA256 against
the source tensor bytes or the retained router prefix. It also verifies auxiliary
files and checks source inventory attributes. Output files include:

- `pruning_manifest.json`: counts, source path and source file attributes.
- `pruning_tensor_manifest.jsonl`: source/output locations, actions and hashes.
- `SHA256SUMS.json`: output shard, copied auxiliary and key metadata hashes.
- `validation_report.json`: tensor count, verified byte count, auxiliary count,
  and separate CPU-routing and NPU-inference flags, initially false.

The hash list excludes itself, `pruning_manifest.json` and
`validation_report.json`. `--verify-only` compares the checkpoint directly with
the source, independently of the stored reports. It does not rewrite a report.
Full source files for removed experts are not hashed. Payload verification checks
stored bytes, without interpreting MXFP arithmetic or running a model forward.

Normal completion publishes the directory by rename. A failed run leaves partial
files for inspection; there is no automatic resume. Confirm the process has
stopped before handling leftover files or locks.

## Validation

Run the synthetic tests without importing the NPU-dependent repository test
initialization:

```bash
python -c 'import runpy; runpy.run_path("tests/ut/tools/test_prune_kimi_k31_a5.py", run_name="__main__")'
```

On 2026-11-12 all 20 tests passed on the Ubuntu server, including symbolic-link
protection. The production source also passed `--plan-only`:

| Plan field | Result |
| --- | ---: |
| Original experts | 896 |
| Retained experts | 32 |
| Source tensors | 498,106 |
| Output tensors | 21,178 |
| Output shards | 337 |
| Output payload bytes | 129,382,401,280 |

Full production checkpoint generation, CPU numerical-routing validation and A5
inference have not been performed with this implementation. These need their own
results before marking the corresponding report flags as verified.

See the [Chinese pruning report](../../Kimi-K3.1-Pro-W4A8-MXFP-E32-A5-Pruning.zh-CN.md)
for the complete procedure and A5 validation requirements.
