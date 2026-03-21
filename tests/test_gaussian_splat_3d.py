"""Tests for the 3D Gaussian splatting front-end."""

import torch

from tinysplat import gaussian_splat_3d, project_gaussians_3d_to_2d


def test_project_gaussian_3d_identity_camera():
    intrinsics = torch.tensor(
        [
            [100.0, 0.0, 32.0],
            [0.0, 120.0, 24.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_to_world = torch.eye(4)

    means = torch.tensor([[0.0, 0.0, 2.0]])
    covariances = torch.eye(3).unsqueeze(0) * 0.01

    projected_means, projected_covariances, depths, visible_mask = project_gaussians_3d_to_2d(
        means=means,
        covariances=covariances,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    assert torch.allclose(projected_means[0], torch.tensor([32.0, 24.0]), atol=1e-5)
    assert depths[0] > 0
    assert visible_mask[0]
    assert projected_covariances.shape == (1, 2, 2)


def test_project_gaussian_3d_camera_to_world_translation():
    intrinsics = torch.tensor(
        [
            [50.0, 0.0, 16.0],
            [0.0, 50.0, 16.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_to_world = torch.eye(4)
    camera_to_world[:3, 3] = torch.tensor([0.0, 0.0, -1.0])

    means = torch.tensor([[0.0, 0.0, 3.0]])
    covariances = torch.eye(3).unsqueeze(0) * 0.01

    _, _, depths, visible_mask = project_gaussians_3d_to_2d(
        means=means,
        covariances=covariances,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    assert visible_mask[0]
    assert torch.allclose(depths, torch.tensor([4.0]), atol=1e-5)


def test_gaussian_splat_3d_output_shape():
    intrinsics = torch.tensor(
        [
            [80.0, 0.0, 8.0],
            [0.0, 80.0, 8.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_to_world = torch.eye(4)

    means = torch.tensor([[0.0, 0.0, 2.0], [0.1, -0.1, 2.5]])
    covariances = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 0.02
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    opacities = torch.tensor([0.8, 0.6])

    image = gaussian_splat_3d(
        means=means,
        covariances=covariances,
        colors=colors,
        opacities=opacities,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        height=16,
        width=16,
        device="cpu",
    )

    assert image.shape == (16, 16, 3)
    assert torch.all(image >= 0)
    assert torch.all(image <= 1)
