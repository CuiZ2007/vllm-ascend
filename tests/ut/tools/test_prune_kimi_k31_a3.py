# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise real safetensors files without importing the inference runtime."""

import copy
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "tools/kimi_k31/prune_experts_a3.py"
SPEC = importlib.util.spec_from_file_location("prune_k31", SCRIPT)
pruner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pruner)


class TestPruneKimiK31(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.config = {
            "text_config": {
                "num_experts": 4,
                "num_experts_per_token": 2,
                "num_expert_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "num_shared_experts": 2,
                "num_hidden_layers": 3,
                "hidden_size": 2,
            }
        }
        self.tensors = {}
        for layer in (1, 2):
            prefix = f"language_model.model.layers.{layer}.block_sparse_moe."
            for expert in range(4):
                for field in pruner.PARAMETERS:
                    self.tensors[f"{prefix}experts.{expert}.{field}"] = ("U8", [2], bytes([expert, layer]))
            self.tensors[prefix + "gate.weight"] = ("BF16", [4, 2], struct.pack("<8H", *([0x3F80] * 8)))
            self.tensors[prefix + "gate.e_score_correction_bias"] = ("F32", [4], struct.pack("<4f", 0, 1, 2, 3))
            self.tensors[prefix + "shared_experts.w1.weight"] = ("U8", [2], b"ab")
        self.tensors["language_model.model.layers.1.self_attn.gate.weight"] = (
            "F32",
            [4],
            struct.pack("<4f", 9, 8, 7, 6),
        )
        self.write_source()

    def write_source(self):
        header, payload = {"__metadata__": {"format": "pt"}}, bytearray()
        for key, (dtype, shape, data) in sorted(self.tensors.items()):
            header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
            payload.extend(data)
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        (self.source / "weights.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        (self.source / "config.json").write_text(json.dumps(self.config))
        (self.source / pruner.INDEX).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": len(payload)},
                    "weight_map": {k: "weights.safetensors" for k in self.tensors},
                }
            )
        )
        (self.source / pruner.QUANT).write_text(json.dumps({**{k: "W4A8" for k in self.tensors}, "group_size": 32}))
        (self.source / "tokenizer.json").write_text('{"fixture":true}')

    def test_full_copy_crops_routers_and_preserves_attention(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        summary = pruner.prune(self.source, self.output, keep=2)
        self.assertEqual(summary["output_tensors"], len(self.tensors) - 2 * 2 * 12)
        new_config = copy.deepcopy(self.config)
        new_config["text_config"]["num_experts"] = 2
        self.assertEqual(pruner.read_json(self.output / "config.json"), new_config)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)
        header, start = pruner.read_header(next(self.output.glob("*.safetensors")))
        data = next(self.output.glob("*.safetensors")).read_bytes()
        for key, desc in header.items():
            if key == "__metadata__":
                continue
            original = self.tensors[key][2]
            if pruner.router_match(key):
                original = original[: len(original) // 2]
            self.assertEqual(data[start + desc["data_offsets"][0] : start + desc["data_offsets"][1]], original)
        self.assertEqual(
            pruner.validate(self.source, self.output, 2)["tensor_bytes_verified"], summary["output_tensors"]
        )

    def test_dry_run_does_not_create_files(self):
        pruner.prune(self.source, self.output, 2, plan_only=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".lock").exists())

    def test_missing_expert_auxiliary_tensor_rejected(self):
        del self.tensors["language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_offset"]
        self.write_source()
        with self.assertRaisesRegex(ValueError, "Missing expert"):
            pruner.prune(self.source, self.output, 2)
        self.assertFalse(self.output.exists())

    def test_missing_correction_bias_rejected(self):
        del self.tensors["language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"]
        self.write_source()
        with self.assertRaisesRegex(ValueError, "Missing router"):
            pruner.prune(self.source, self.output, 2)

    def test_output_and_lock_are_not_overwritten(self):
        for suffix in ("", ".partial", ".lock"):
            with self.subTest(suffix=suffix):
                path = self.root / ("output" + suffix)
                path.touch()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    pruner.prune(self.source, self.output, 2)
                path.unlink()

    def test_invalid_keep_and_nested_output_rejected(self):
        for keep in (1, 4, 5):
            with self.assertRaises(ValueError):
                pruner.prune(self.source, self.output, keep)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            pruner.prune(self.source, self.source / "output", 2)

    def test_corruption_detected_by_source_comparison(self):
        pruner.prune(self.source, self.output, 2)
        path = next(self.output.glob("*.safetensors"))
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            pruner.validate(self.source, self.output, 2)

    def test_quantization_index_mismatch_rejected(self):
        (self.source / pruner.QUANT).write_text('{"group_size":32}')
        with self.assertRaisesRegex(ValueError, "Quantization keys"):
            pruner.prune(self.source, self.output, 2)

    def test_header_overlap_rejected(self):
        path = self.source / "weights.safetensors"
        data = path.read_bytes()
        size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + size])
        key = next(k for k in header if k != "__metadata__")
        header[key]["data_offsets"] = [1, 3]
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + data[8 + size :])
        with self.assertRaisesRegex(ValueError, "Gap or overlap"):
            pruner.prune(self.source, self.output, 2)


if __name__ == "__main__":
    unittest.main()
