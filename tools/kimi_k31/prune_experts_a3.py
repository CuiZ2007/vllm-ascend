# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Stream a Kimi-K3.1 W4A8 checkpoint into a prefix-expert checkpoint.

Only the standard library is required. No dequantization or NPU operations occur.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path

PARAMETERS = frozenset(
    f"{w}.{p}" for w in ("w1", "w2", "w3") for p in ("weight", "weight_scale", "weight_offset", "scale_bias")
)
GLOBAL_KEYS = frozenset(("group_size", "metadata", "model_quant_type", "optional", "version"))
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
}
CHUNK_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 100 * 1024 * 1024
DISK_MARGIN_BYTES = 64 * 1024 * 1024
METADATA_SPACE_MULTIPLIER = 4
REPORTS = frozenset(
    ("pruning_manifest.json", "pruning_tensor_manifest.jsonl", "SHA256SUMS.json", "validation_report.json")
)
INDEX = "quant_model_weights.safetensors.index.json"
QUANT = "quant_model_description.json"


def expert_match(key):
    parts = key.split(".")
    if (
        len(parts) >= 8
        and parts[:3] == ["language_model", "model", "layers"]
        and parts[3].isdecimal()
        and parts[4:6] == ["block_sparse_moe", "experts"]
        and parts[6].isdecimal()
    ):
        return None, parts[3], parts[6], ".".join(parts[7:])
    return None


def router_match(key):
    parts = key.split(".")
    if (
        len(parts) == 7
        and parts[:3] == ["language_model", "model", "layers"]
        and parts[3].isdecimal()
        and parts[4:6] == ["block_sparse_moe", "gate"]
        and parts[6] in ("weight", "e_score_correction_bias")
    ):
        return None, parts[3], parts[6]
    return None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlinks are unsupported: {path}")
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_mode)
    return result


def read_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= MAX_HEADER_BYTES, f"Invalid header length: {path}")
        raw = stream.read(length)
        require(len(raw) == length, f"Truncated header: {path}")
    header = json.loads(raw, object_pairs_hook=unique_pairs)
    require(isinstance(header, dict), f"Invalid header: {path}")
    spans = []
    for key, desc in header.items():
        if key == "__metadata__":
            require(isinstance(desc, dict) and all(isinstance(v, str) for v in desc.values()), "Invalid metadata")
            continue
        shape, offsets, dtype = desc["shape"], desc["data_offsets"], desc["dtype"]
        require(all(type(x) is int and x >= 0 for x in shape), f"Invalid shape: {key}")
        require(dtype in DTYPE_BYTES, f"Unsupported dtype {dtype}: {key}")
        require(len(offsets) == 2 and all(type(x) is int for x in offsets), f"Invalid offsets: {key}")
        start, end = offsets
        require(
            0 <= start <= end and end - start == math.prod(shape) * DTYPE_BYTES[dtype], f"Invalid byte count: {key}"
        )
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor, f"Gap or overlap: {path}")
        cursor = end
    require(8 + length + cursor == path.stat().st_size, f"Incorrect file size: {path}")
    return header, 8 + length


def build_plan(source, keep):
    require(not (set(inventory(source)) & REPORTS), "Source already has pruning reports")
    # Check every original expert before dropping any payload, including removed experts.
    config, index, quant = (read_json(source / name) for name in ("config.json", INDEX, QUANT))
    cfg = config["text_config"]
    count = cfg["num_experts"]
    require(
        type(keep) is int and cfg["num_experts_per_token"] <= keep < count, "Require top-k <= keep < original experts"
    )
    require(cfg["num_expert_group"] == cfg["topk_group"] == 1, "Only single-group routing is supported")
    require(cfg["first_k_dense_replace"] == cfg["moe_layer_freq"] == 1, "Unsupported MoE layer layout")
    require(cfg["num_shared_experts"] == 2, "Expected two shared experts")
    expected_layers = set(range(1, cfg["num_hidden_layers"]))
    require(bool(expected_layers), "No MoE layers")
    weight_map = index["weight_map"]
    require(bool(weight_map), "Empty weight index")
    shard_names = set(weight_map.values())
    require(
        all(Path(n).name == n and n.endswith(".safetensors") and "\\" not in n for n in shard_names),
        "Unsafe shard path",
    )
    require(shard_names == {p.name for p in source.glob("*.safetensors")}, "Shard set differs from index")
    seen, expert_fields, routers, plans = set(), {}, set(), []
    total = 0
    for name in sorted(shard_names):
        header, data_start = read_header(source / name)
        retained = []
        for key, desc in header.items():
            if key == "__metadata__":
                continue
            require(key not in seen and weight_map.get(key) == name, f"Index mismatch: {key}")
            seen.add(key)
            length = desc["data_offsets"][1] - desc["data_offsets"][0]
            total += length
            expert, router = expert_match(key), router_match(key)
            require(".block_sparse_moe.experts." not in key or expert is not None, f"Unsupported expert name: {key}")
            require(".block_sparse_moe.gate." not in key or router is not None, f"Unsupported router name: {key}")
            output_desc = copy.deepcopy(desc)
            if expert:
                layer, expert_id = int(expert[1]), int(expert[2])
                require(layer in expected_layers and 0 <= expert_id < count, f"Unexpected expert: {key}")
                require(expert[3] in PARAMETERS, f"Unexpected expert parameter: {key}")
                expert_fields.setdefault((layer, expert_id), set()).add(expert[3])
                if expert_id >= keep:
                    continue
            if router:
                layer, kind = int(router[1]), router[2]
                require(layer in expected_layers, f"Unexpected router: {key}")
                shape = [count, cfg["hidden_size"]] if kind == "weight" else [count]
                require(
                    desc["shape"] == shape and desc["dtype"] == ("BF16" if kind == "weight" else "F32"),
                    f"Invalid router: {key}",
                )
                routers.add((layer, kind))
                output_desc["shape"][0] = keep
                length = length // count * keep
            retained.append(
                {
                    "key": key,
                    "source_file": name,
                    "source_offset": data_start + desc["data_offsets"][0],
                    "length": length,
                    "descriptor": output_desc,
                    "action": "crop_router_rows" if router else "copy_exact",
                }
            )
        if retained:
            plans.append({"metadata": header.get("__metadata__"), "tensors": retained})
    require(seen == set(weight_map), "Tensor set differs from index")
    require(total == index["metadata"]["total_size"], "Index total_size mismatch")
    require(
        len(expert_fields) == len(expected_layers) * count and all(v == PARAMETERS for v in expert_fields.values()),
        "Missing expert tensors",
    )
    require(
        routers == {(layer, kind) for layer in expected_layers for kind in ("weight", "e_score_correction_bias")},
        "Missing router tensors",
    )
    require(set(quant) - GLOBAL_KEYS == seen, "Quantization keys differ from index")
    output_keys = {t["key"] for p in plans for t in p["tensors"]}
    new_config = copy.deepcopy(config)
    new_config["text_config"]["num_experts"] = keep
    new_quant = {k: v for k, v in quant.items() if k in GLOBAL_KEYS or k in output_keys}
    return {
        "config": new_config,
        "quant": new_quant,
        "plans": plans,
        "summary": {
            "original_experts": count,
            "keep": keep,
            "source_tensors": len(seen),
            "output_tensors": len(output_keys),
            "output_shards": len(plans),
            "output_bytes": sum(t["length"] for p in plans for t in p["tensors"]),
        },
    }


def transfer(stream, offset, length, output=None):
    stream.seek(offset)
    digest = hashlib.sha256()
    while length:
        block = stream.read(min(length, CHUNK_BYTES))
        require(bool(block), "Unexpected end of tensor data")
        digest.update(block)
        if output is not None:
            output.write(block)
        length -= len(block)
    return digest.hexdigest()


def validate(source, checkpoint, keep):
    """Independently re-plan and compare all output bytes to source slices."""
    plan = build_plan(source, keep)
    require(read_json(checkpoint / "config.json") == plan["config"], "Config mismatch")
    require(read_json(checkpoint / QUANT) == plan["quant"], "Quantization mismatch")
    index = read_json(checkpoint / INDEX)
    expected = {t["key"]: t for p in plan["plans"] for t in p["tensors"]}
    require(set(index["weight_map"]) == set(expected), "Output index keys mismatch")
    require(index["metadata"]["total_size"] == plan["summary"]["output_bytes"], "Output total_size mismatch")
    seen = set()
    shards = set(index["weight_map"].values())
    require(shards == {p.name for p in checkpoint.glob("*.safetensors")}, "Output shard set mismatch")
    for name in sorted(shards):
        require(Path(name).name == name and "\\" not in name, "Unsafe output shard path")
        header, data_start = read_header(checkpoint / name)
        with (checkpoint / name).open("rb") as output:
            for key, desc in header.items():
                if key == "__metadata__":
                    continue
                require(
                    key in expected and key not in seen and index["weight_map"].get(key) == name,
                    f"Unexpected output tensor: {key}",
                )
                seen.add(key)
                tensor = expected[key]
                require(
                    desc["shape"] == tensor["descriptor"]["shape"] and desc["dtype"] == tensor["descriptor"]["dtype"],
                    f"Descriptor mismatch: {key}",
                )
                with (source / tensor["source_file"]).open("rb") as original:
                    digest = transfer(original, tensor["source_offset"], tensor["length"])
                require(
                    digest == transfer(output, data_start + desc["data_offsets"][0], tensor["length"]),
                    f"Tensor hash mismatch: {key}",
                )
    require(seen == set(expected), "Missing output tensors")
    auxiliary = auxiliary_files(inventory(source))
    for rel, identity in auxiliary.items():
        path = checkpoint / rel
        require(path.is_file() and not path.is_symlink(), f"Missing auxiliary file: {rel}")
        require(path.stat().st_size == identity[0], f"Auxiliary size mismatch: {rel}")
        with (source / rel).open("rb") as original, path.open("rb") as copied:
            require(
                transfer(original, 0, identity[0]) == transfer(copied, 0, identity[0]),
                f"Auxiliary file mismatch: {rel}",
            )
    checksums = checkpoint / "SHA256SUMS.json"
    require(checksums.is_file() and not checksums.is_symlink(), "Missing checksum manifest")
    hashes = read_json(checksums)
    required = set(shards) | set(auxiliary) | {"config.json", INDEX, QUANT, "pruning_tensor_manifest.jsonl"}
    require(set(hashes) == required, "Checksum file set mismatch")
    for rel, expected_hash in hashes.items():
        path = checkpoint / rel
        require(path.is_file() and not path.is_symlink(), f"Missing checksummed file: {rel}")
        with path.open("rb") as stream:
            require(transfer(stream, 0, path.stat().st_size) == expected_hash, f"File checksum mismatch: {rel}")
    return {
        "tensors_verified": len(seen),
        "tensor_bytes_verified": plan["summary"]["output_bytes"],
        "auxiliary_files_verified": len(auxiliary),
        "npu_inference_verified": False,
    }


def auxiliary_files(files):
    reserved = REPORTS | {"config.json", INDEX, QUANT}
    return {
        rel: identity for rel, identity in files.items() if rel not in reserved and not rel.endswith(".safetensors")
    }


def required_disk_bytes(plan, files):
    # Include headers, copied assets, JSON metadata and reports. Keep a conservative
    # margin for manifests, filesystem allocation and temporary metadata.
    metadata_bytes = len(json.dumps(plan, ensure_ascii=False).encode())
    auxiliary_bytes = sum(identity[0] for identity in auxiliary_files(files).values())
    return (
        plan["summary"]["output_bytes"]
        + auxiliary_bytes
        + metadata_bytes * METADATA_SPACE_MULTIPLIER
        + DISK_MARGIN_BYTES
    )


def prune(source, output, keep=32, plan_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source.is_dir(), "Source directory does not exist")
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Source/output directories must be disjoint",
    )
    partial, lock = output.with_name(output.name + ".partial"), output.with_name(output.name + ".lock")
    require(not any(p.exists() for p in (output, partial, lock)), "Output, partial directory or lock already exists")
    before = inventory(source)
    plan = build_plan(source, keep)
    if plan_only:
        require(inventory(source) == before, "Source changed while planning")
        return plan["summary"]
    require(output.parent.is_dir(), "Output parent must exist")
    require(shutil.disk_usage(output.parent).free >= required_disk_bytes(plan, before), "Insufficient disk space")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        require(not output.exists(), "Output appeared during planning")
        partial.mkdir()
        weight_map, hashes = {}, {}
        count = len(plan["plans"])
        with (partial / "pruning_tensor_manifest.jsonl").open("x", encoding="utf-8") as manifest:
            for number, shard in enumerate(plan["plans"], 1):
                name = f"quant_model_weights-{number:05d}-of-{count:05d}.safetensors"
                header, offset = {}, 0
                if shard["metadata"] is not None:
                    header["__metadata__"] = shard["metadata"]
                for tensor in shard["tensors"]:
                    desc = copy.deepcopy(tensor["descriptor"])
                    desc["data_offsets"] = [offset, offset + tensor["length"]]
                    header[tensor["key"]] = desc
                    weight_map[tensor["key"]] = name
                    offset += tensor["length"]
                raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
                raw += b" " * (-len(raw) % 8)
                temporary = partial / (name + ".tmp")
                with temporary.open("xb") as destination:
                    destination.write(struct.pack("<Q", len(raw)) + raw)
                    # One input file per output shard, with bounded data buffers.
                    with (source / shard["tensors"][0]["source_file"]).open("rb") as original:
                        for tensor in shard["tensors"]:
                            digest = transfer(original, tensor["source_offset"], tensor["length"], destination)
                            record = {k: v for k, v in tensor.items() if k != "descriptor"}
                            manifest.write(json.dumps({**record, "output_file": name, "sha256": digest}) + "\n")
                    destination.flush()
                    os.fsync(destination.fileno())
                temporary.rename(partial / name)
                with (partial / name).open("rb") as stream:
                    hashes[name] = transfer(stream, 0, (partial / name).stat().st_size)
        for rel in auxiliary_files(before):
            dest = partial / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dest)
            with (source / rel).open("rb") as original, dest.open("rb") as copied:
                digest = transfer(original, 0, before[rel][0])
                require(digest == transfer(copied, 0, before[rel][0]), f"Auxiliary file mismatch: {rel}")
                hashes[rel] = digest
        write_json(partial / "config.json", plan["config"])
        write_json(partial / QUANT, plan["quant"])
        write_json(
            partial / INDEX, {"metadata": {"total_size": plan["summary"]["output_bytes"]}, "weight_map": weight_map}
        )
        require(inventory(source) == before, "Source changed during generation")
        write_json(
            partial / "pruning_manifest.json", {**plan["summary"], "source": str(source), "source_files": before}
        )
        for name in ("config.json", INDEX, QUANT, "pruning_tensor_manifest.jsonl"):
            with (partial / name).open("rb") as stream:
                hashes[name] = transfer(stream, 0, (partial / name).stat().st_size)
        write_json(partial / "SHA256SUMS.json", hashes)
        report = validate(source, partial, keep)
        require(inventory(source) == before, "Source changed during validation")
        write_json(partial / "validation_report.json", report)
        require(not output.exists(), "Output appeared during generation")
        partial.rename(output)
    finally:
        lock.unlink()
    return plan["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Read and compare an existing output to the source")
    args = parser.parse_args()
    require(not (args.plan_only and args.verify_only), "Choose only one validation mode")
    result = (
        validate(args.source.resolve(), args.output.resolve(), args.keep)
        if args.verify_only
        else prune(args.source, args.output, args.keep, args.plan_only)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
