# SPDX-License-Identifier: Apache-2.0
# CPU byte-preserving pruning for the local A3 host. No dtype conversion.
import argparse
import collections
import json
from pathlib import Path

from tools.kimi_k3 import prune_experts_a5 as core

WEIGHTS = Path("/data/c30087248/weights")


def validate_target(keep, ep):
    if ep != 16 or keep < 16 or keep % ep:
        raise ValueError("A3 entry requires EP16 and keep-experts >=16 divisible by 16")


def quantization_report(src):
    q = core.read_json(Path(src) / "quant_model_description.json")
    counts = dict(collections.Counter(v for v in q.values() if isinstance(v, str)))
    return {
        "target": "A3/EP16",
        "quantization_entries": counts,
        "preserves_original_dtype": True,
        "inference_validated": False,
        "fp8_runtime_blocker_observed": "W8A8_MXFP8" in counts,
        "note": "Current A3 v0.30.0 format-cast rejects FP8 E4M3FN. Pruning does not fix it.",
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Local A3 CPU pruning; not requantization or inference.")
    ap.add_argument("--src", default=str(WEIGHTS / "Eco-Tech/Kimi-K3-w4a8-mxfp-QuaRot"))
    ap.add_argument("--dst", default=str(WEIGHTS / "Kimi-K3-w4a8-mxfp-QuaRot-A3-experts256"))
    ap.add_argument("--work", default="/data/c30087248/pruning-runs/kimi-k3-a3-ep256")
    ap.add_argument("--keep-experts", type=int, default=256)
    ap.add_argument("--ep-size", type=int, default=16)
    ap.add_argument("--mode", choices=["plan", "run"], default="plan")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    validate_target(args.keep_experts, args.ep_size)
    if args.workers < 1:
        ap.error("--workers must be positive")
    if args.mode == "run" and args.keep_experts != 256:
        ap.error("Original production writer supports only 256 experts")
    return args


def main():
    args = parse_args()
    report = quantization_report(args.src)
    print(json.dumps(report), flush=True)
    core.run(args)
    core.atomic_json(Path(args.work) / "a3_quantization_report.json", report)


if __name__ == "__main__":
    main()
