# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

from tools.kimi_k3 import prune_experts_a3 as a3


class A3PruneTests(unittest.TestCase):
    def test_ep16_rejects_ep8_only_target(self):
        a3.validate_target(256, 16)
        with self.assertRaises(ValueError):
            a3.validate_target(24, 16)

    def test_fp8_report_preserves_dtype_and_does_not_claim_runtime_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "quant_model_description.json"
            p.write_text(json.dumps({"dense.weight": "W8A8_MXFP8", "expert.weight": "W4A8_MXFP"}))
            r = a3.quantization_report(d)
            self.assertTrue(r["fp8_runtime_blocker_observed"])
            self.assertTrue(r["preserves_original_dtype"])
            self.assertFalse(r["inference_validated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
