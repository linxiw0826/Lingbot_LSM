import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from probe_wan_runtime import (
    CleanT0Unsupported, causal_latent_index, map_fixture_frames,
    direct_forward_conditioning, raise_runtime_stage_error, runtime_decision,
    token_slice_for_frame,
)


class RuntimeProbePureTests(unittest.TestCase):
    def test_conditioning_y_keeps_channel_axis_and_is_wrapped_once(self):
        import torch

        y = torch.zeros(20, 21, 58, 104)
        context_item = torch.zeros(3, 4)
        plucker = torch.zeros(1, 6, 21, 58, 104)
        prepared = {
            "y": y,
            "context": [context_item],
            "dit_cond_dict": {"c2ws_plucker_emb": (plucker,)},
        }

        forward = direct_forward_conditioning(prepared)

        self.assertEqual(tuple(prepared["y"].shape), (20, 21, 58, 104))
        self.assertIs(forward["context"], prepared["context"])
        self.assertEqual(len(forward["y"]), 1)
        self.assertIs(forward["y"][0], y)
        self.assertEqual(tuple(forward["y"][0].shape), (20, 21, 58, 104))
        self.assertIs(forward["dit_cond_dict"], prepared["dit_cond_dict"])

    def test_causal_mapping_for_frozen_target(self):
        self.assertEqual(causal_latent_index(70, 4, 21), 18)
        self.assertEqual(causal_latent_index(0, 4, 21), 0)
        self.assertEqual(causal_latent_index(1, 4, 21), 1)

    def test_target_token_slice(self):
        value = token_slice_for_frame(
            70, temporal_stride=4, latent_frames=21,
            latent_h=58, latent_w=104, patch_size=(1, 2, 2),
        )
        self.assertEqual(value["latent_t"], 18)
        self.assertEqual(value["token_count"], 1508)
        self.assertEqual((value["token_start"], value["token_end"]), (27144, 28652))

    def test_fail_closed_decision(self):
        self.assertEqual(runtime_decision(False, False, None), "BLOCKED_STATIC_FACTS")
        self.assertEqual(runtime_decision(True, True, None), "FIXTURE_FREEZE_PASS")
        self.assertEqual(runtime_decision(True, False, "BLOCKED_GPU_RUNTIME"), "BLOCKED_GPU_RUNTIME")

    def test_full_support_mapping_and_many_to_one_dedup(self):
        fixture = {
            "target_full_frame": 12,
            "support_full_half_open": [10, 15],
            "planner_windows": [{
                "window_index": 2, "owned_half_open": [10, 15],
                "source_frame_index": list(range(81)),
            }],
        }
        value = map_fixture_frames(
            fixture, temporal_stride=4, latent_frames=21,
            latent_h=58, latent_w=104, patch_size=(1, 2, 2),
            include_support=True,
        )
        self.assertTrue(value["complete"])
        self.assertEqual(len(value["per_frame"]), 5)
        groups = value["deduplicated_many_to_one_groups"]
        self.assertEqual([group["full_frames"] for group in groups], [[10, 11, 12], [13, 14]])
        self.assertEqual(value["target"]["latent_t"], 3)

    def test_target_only_mapping_for_nontrain_fixture(self):
        fixture = {
            "target_full_frame": 199,
            "support_full_half_open": [176, 224],
            "planner_windows": [{
                "window_index": 3, "owned_half_open": [176, 224],
                "source_frame_index": list(range(168, 249)),
            }],
        }
        value = map_fixture_frames(
            fixture, temporal_stride=4, latent_frames=21,
            latent_h=58, latent_w=104, patch_size=(1, 2, 2),
            include_support=False,
        )
        self.assertEqual(len(value["per_frame"]), 1)
        self.assertEqual(value["target"]["local_frame"], 31)
        self.assertEqual(value["target"]["latent_t"], 8)

    def test_timestep_specific_error_is_clean_t0_blocker(self):
        with self.assertRaises(CleanT0Unsupported):
            raise_runtime_stage_error("selection", ValueError("unsupported timestep t=0"))

    def test_generic_forward_error_is_gpu_runtime_blocker(self):
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_GPU_RUNTIME"):
            raise_runtime_stage_error("forward", RuntimeError("matrix shape mismatch"))

    def test_missing_call_signature_is_minimal_api_blocker(self):
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_MINIMAL_FORWARD_API_MISSING"):
            raise_runtime_stage_error("selection", TypeError("unexpected keyword argument"))

    def test_shell_runner_is_nonfatal_on_setup_failure(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ)
            env.update({
                "REPO": str(Path(temporary) / "missing_repo"),
                "CASES": str(Path(temporary) / "missing_cases"),
                "CKPT": str(Path(temporary) / "missing_ckpt"),
                "PROBE_OUT": str(Path(temporary) / "out"),
            })
            result = subprocess.run(
                ["bash", str(root / "run_probe.sh")], env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STATIC_PROBE_PYTHON_EXIT=", result.stdout)
            self.assertIn("RUNTIME_PROBE_PYTHON_EXIT=", result.stdout)


if __name__ == "__main__":
    unittest.main()
