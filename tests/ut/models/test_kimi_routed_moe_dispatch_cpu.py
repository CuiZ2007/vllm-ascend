# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real MoE constructor without loading the NPU backend.

Run with --confcutdir=tests/ut/models. Kernel numerics need separate NPU tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


class FakeReplicatedLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = kwargs.get("quant_config")


class FakeOutputTransform(nn.Module):
    def __init__(self, norm, up_proj):
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj


@pytest.mark.parametrize("latent_size", [None, 2])
@pytest.mark.parametrize("quantized", [False, True])
def test_moe_constructs_routed_projections_with_eager_dispatch(latent_size, quantized):
    tree = ast.parse((ROOT / "vllm_ascend/models/kimi_k3/kimi_k3.py").read_text())
    classes = []
    for name in ("AscendKimiK3RoutedLinear", "AscendKimiMoE"):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
        classes.append(cls)

    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        module = nn.Module()
        module.moe_config = SimpleNamespace(num_logical_experts=256, num_experts=256, num_local_experts=32)
        return module

    scope = dict(
        nn=nn,
        torch=torch,
        ReplicatedLinear=FakeReplicatedLinear,
        GateLinear=lambda input_size, output_size, **kwargs: nn.Linear(input_size, output_size),
        AscendKimiRoutedOutputTransform=FakeOutputTransform,
        FusedMoEFactory=factory,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])), "kimi_k3.py", "exec"),
        scope,
    )
    config = SimpleNamespace(
        hidden_size=4,
        moe_intermediate_size=8,
        num_experts=256,
        num_experts_per_token=2,
        routed_expert_hidden_size=latent_size,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        num_shared_experts=None,
        hidden_act="silu",
        moe_renormalize=True,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
    )
    quant_config = SimpleNamespace(get_name=lambda: "ascend") if quantized else None
    parallel = SimpleNamespace(enable_eplb=False)
    moe = scope["AscendKimiMoE"](
        config=config, vllm_config=SimpleNamespace(parallel_config=parallel), quant_config=quant_config
    )
    call = factory_calls[0]
    if latent_size is None:
        assert call["routed_input_transform"] is None
        assert call["routed_output_transform"] is None
        return

    for layer in (moe.routed_expert_down_proj, moe.routed_expert_up_proj):
        assert isinstance(layer, scope["AscendKimiK3RoutedLinear"])
        assert layer._single_group_gmm_nz is True
        assert layer.quant_config is quant_config
    assert call["routed_input_transform"] is moe.routed_expert_down_proj
    assert call["routed_output_transform"].up_proj is moe.routed_expert_up_proj
    assert (moe.routed_expert_down_proj.input_size, moe.routed_expert_down_proj.output_size) == (4, 2)
    assert (moe.routed_expert_up_proj.input_size, moe.routed_expert_up_proj.output_size) == (2, 4)
