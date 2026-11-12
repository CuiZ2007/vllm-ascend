# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the actual FlashMLA MLAPO admission and staging allocation."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


class QuantizedProjection:
    pass


def _implementation():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(source.read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    names = {"_supports_flashmla_mlapo", "_can_use_flashmla_mlapo", "_prepare_flashmla_writer_buffers"}
    methods: list[ast.stmt] = [node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(
        torch=torch,
        cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
        PROLOG_V3_SUPPORTED_INPUT_SIZES=(7168,),
        FLASHMLA_BLOCK_SIZE=128,
        MLAPO_MAX_SUPPORTED_TOKENS=256,
        MLAPO_MAX_CACHE_BLOCK_SIZE=1024,
        MLAPO_CACHE_BLOCK_ALIGNMENT=16,
        UnquantizedLinearMethod=type("UnquantizedLinearMethod", (), {}),
        AscendW8A8MXFP8DynamicLinearMethod=QuantizedProjection,
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    obj = SimpleNamespace(
        use_flashmla=True,
        enable_mlapo=True,
        is_draft_model=False,
        use_mla_rope=False,
        support_fp8_attention=True,
        fa_quant_layer=False,
        enable_kv_nz=False,
        use_pcp=False,
        use_dcp=False,
        is_pcp_decode_sharded=False,
        _mlapo_uses_native_weights=False,
        mlapo_weight_quant_mode=3,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        W_UV=torch.empty(0),
        _flashmla_writer_buffers=None,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16), cache_config=SimpleNamespace(block_size=128)
        ),
        fused_qkv_a_proj=SimpleNamespace(input_size=7168, quant_method=QuantizedProjection()),
        q_proj=SimpleNamespace(quant_method=QuantizedProjection()),
        supports_prolog_v3_dtype=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        _get_context_prolog_quant_method=lambda layer: layer.quant_method,
    )
    for name in names:
        setattr(obj, name, MethodType(namespace[name], obj))
    return obj


def _inputs():
    cache = torch.empty((1, 128, 1, 576), dtype=torch.bfloat16)
    return (cache[..., :512], cache[..., 512:]), SimpleNamespace(
        num_decode_tokens=1, num_prefills=0, external_flashmla=object()
    )


def test_flashmla_decode_requires_preallocated_writer():
    impl = _implementation()
    cache, metadata = _inputs()
    assert not impl._can_use_flashmla_mlapo(cache, metadata)
    impl._prepare_flashmla_writer_buffers()
    original = impl._flashmla_writer_buffers
    assert original[0].shape == (1, 128, 1, 512)
    assert original[1].shape == (1, 128, 1, 64)
    assert original[2].numel() == 16
    assert impl._can_use_flashmla_mlapo(cache, metadata)
    impl._prepare_flashmla_writer_buffers()
    assert impl._flashmla_writer_buffers is original
    metadata.num_decode_tokens = 17
    assert not impl._can_use_flashmla_mlapo(cache, metadata)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("use_flashmla", False),
        ("is_draft_model", True),
        ("use_mla_rope", True),
        ("use_pcp", True),
        ("is_pcp_decode_sharded", True),
        ("fa_quant_layer", True),
        ("enable_kv_nz", True),
        ("mlapo_weight_quant_mode", 2),
    ],
)
def test_flashmla_unsupported_modes_keep_fallback(attribute, value):
    impl = _implementation()
    impl._prepare_flashmla_writer_buffers()
    setattr(impl, attribute, value)
    assert not impl._can_use_flashmla_mlapo(*_inputs())


def test_flashmla_mixed_prefill_and_incompatible_cache_keep_fallback():
    impl = _implementation()
    impl._prepare_flashmla_writer_buffers()
    cache, metadata = _inputs()
    metadata.num_prefills = 1
    assert not impl._can_use_flashmla_mlapo(cache, metadata)
    metadata.num_prefills = 0
    assert not impl._can_use_flashmla_mlapo(tuple(t.float() for t in cache), metadata)
