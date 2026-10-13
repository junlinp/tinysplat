"""Regression checks for the Python quality controls (run with unittest)."""
import contextlib
import io
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_script("quality_trainer", "train_3d_gaussians_json.py")
viewer = load_script("quality_viewer", "view_gaussians_viser.py")


class QualityControlsTest(unittest.TestCase):
    def test_training_defaults_and_opt_out(self):
        with patch.object(sys, "argv", ["train", "dataset.json"]):
            args = trainer.parse_args()
        self.assertTrue(args.fastgs_late_prune)
        self.assertEqual(args.checkpoint_every, 0)
        with patch.object(sys, "argv", ["train", "dataset.json", "--no-fastgs-late-prune",
                                         "--checkpoint-every", "5000"]):
            args = trainer.parse_args()
        self.assertFalse(args.fastgs_late_prune)
        self.assertEqual(args.checkpoint_every, 5000)

    def test_negative_limits_are_rejected(self):
        for module, argv in [
            (trainer, ["train", "dataset.json", "--checkpoint-every", "-1"]),
            (viewer, ["view", "gaussians.pt", "--gpu-memory-limit-gb", "-1"]),
            (viewer, ["view", "gaussians.pt", "--torch-num-threads", "-1"]),
            (viewer, ["view", "gaussians.pt", "--gpu-memory-limit-gb", "nan"]),
            (viewer, ["view", "gaussians.pt", "--gpu-memory-limit-gb", "inf"]),
        ]:
            with self.subTest(argv=argv), patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    module.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_webgl_export_keeps_faint_gaussians_and_order(self):
        model = SimpleNamespace(
            means=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            sh_coeffs=None, colors=torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
            visible_opacities=lambda: torch.tensor([.001, .8]),
            covariance_matrices=lambda: torch.eye(3).repeat(2, 1, 1),
        )
        arrays = viewer.splat_arrays(model, opacity_floor=0., max_splats=0)
        self.assertEqual(arrays["kept"], 2)
        np.testing.assert_array_equal(arrays["indices"], [0, 1])
        limited = viewer.splat_arrays(model, opacity_floor=0., max_splats=1)
        np.testing.assert_array_equal(limited["indices"], [1])

    def test_calibrated_render_uses_exact_intrinsics_and_resolution(self):
        class Model:
            def render(self, **kwargs):
                self.received = kwargs
                return torch.zeros((7, 5, 3))
        model = Model()
        frame = SimpleNamespace(width=5, height=7,
            intrinsics=torch.tensor([[4., 0., 1.2], [0., 6., 3.1], [0., 0., 1.]]),
            camera_to_world=torch.eye(4))
        rgb, _ = viewer.Renderer(model).render_frame(frame)
        self.assertIs(model.received["intrinsics"], frame.intrinsics)
        self.assertIs(model.received["camera_to_world"], frame.camera_to_world)
        self.assertEqual((model.received["width"], model.received["height"]), (5, 7))
        self.assertEqual(rgb.shape, (7, 5, 3))

    def test_letterbox_preserves_all_pixels(self):
        rgb = np.arange(7 * 5 * 3, dtype=np.uint8).reshape(7, 5, 3)
        for aspect in (2., .4):
            canvas = viewer.letterbox_image(rgb, aspect)
            h, w = canvas.shape[:2]
            y, x = (h - 7) // 2, (w - 5) // 2
            np.testing.assert_array_equal(canvas[y:y + 7, x:x + 5], rgb)
            self.assertLessEqual(abs(w / h - aspect), 1 / h)

    def test_recorded_camera_preserves_roll_and_fov(self):
        # A rolled optical camera must not be forced to world +Z up.
        pose = torch.tensor([[0., -1., 0., 2.], [1., 0., 0., 3.],
                             [0., 0., 1., 4.], [0., 0., 0., 1.]])
        frame = SimpleNamespace(camera_to_world=pose, height=640,
                                intrinsics=torch.tensor([[305., 0., 272.],
                                                         [0., 305., 320.],
                                                         [0., 0., 1.]]))
        client = SimpleNamespace(camera=SimpleNamespace())
        viewer.set_camera_to_frame(client, frame, look_distance=2.)
        np.testing.assert_allclose(client.camera.position, [2., 3., 4.])
        np.testing.assert_allclose(client.camera.look_at, [2., 3., 6.])
        np.testing.assert_allclose(client.camera.up_direction, [1., 0., 0.])
        self.assertAlmostEqual(client.camera.fov, 2 * math.atan2(320., 305.))


if __name__ == "__main__":
    unittest.main()
