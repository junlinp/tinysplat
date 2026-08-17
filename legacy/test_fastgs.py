"""Unit tests for FastGS VCD/VCP densify and prune helpers."""

import math

import torch

from tinysplat.fastgs import (
    FastGSConfig,
    accumulate_vcd_scores,
    densify_mask_vcd,
    prune_mask_fastgs_densify,
    prune_mask_vcp,
    split_child_log_scales,
)


def test_split_child_log_scales_multiplies_when_shrink_below_one():
    log_scales = torch.zeros(2, 3)
    out = split_child_log_scales(log_scales, 0.8)
    assert torch.allclose(out.exp(), torch.full_like(out, 0.8))


def test_split_child_log_scales_divides_when_shrink_above_one():
    log_scales = torch.zeros(1, 3)
    out = split_child_log_scales(log_scales, 1.6)
    assert torch.allclose(out.exp(), torch.full_like(out, 1.0 / 1.6))


def test_vcd_floor_mean():
    a = torch.tensor([10.0, 4.0, 0.0])
    b = torch.tensor([11.0, 5.0, 1.0])
    out = accumulate_vcd_scores([a, b])
    # floor((10+11)/2)=10, floor(9/2)=4, floor(1/2)=0
    assert out.tolist() == [10.0, 4.0, 0.0]


def test_vcd_clone_small_split_large():
    cfg = FastGSConfig(densify_score_thresh=5.0, grad_thresh=0.0002, grow_scale3d=0.01)
    scores = torch.tensor([6.0, 6.0, 1.0])
    grads = torch.tensor([0.001, 0.001, 0.001])
    log_scales = torch.log(
        torch.tensor([[0.005, 0.005, 0.005], [0.05, 0.05, 0.05], [0.005, 0.005, 0.005]])
    )
    clone, split = densify_mask_vcd(scores, grads, log_scales, cfg, scene_scale=1.0)
    assert clone.tolist() == [True, False, False]
    assert split.tolist() == [False, True, False]


def test_vcd_uses_absgrad_thresh_for_split_only():
    cfg = FastGSConfig(
        densify_score_thresh=5.0,
        grad_thresh=0.0002,
        grad_abs_thresh=0.0009,
        absgrad_split=True,
        grow_scale3d=0.01,
    )
    scores = torch.tensor([6.0, 6.0, 6.0])
    grads = torch.tensor([0.001, 0.0001, 0.001])
    abs_grads = torch.tensor([0.0001, 0.001, 0.0005])
    log_scales = torch.log(
        torch.tensor(
            [
                [0.005, 0.005, 0.005],
                [0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05],
            ]
        )
    )
    clone, split = densify_mask_vcd(
        scores, grads, log_scales, cfg, scene_scale=1.0, abs_grad_norms=abs_grads
    )
    # small + vanilla grad → clone; large + absgrad>=0.0009 → split; large + 0.0005 → neither
    assert clone.tolist() == [True, False, False]
    assert split.tolist() == [False, True, False]


def test_densify_prune_is_half_of_low_opacity_not_vcp():
    cfg = FastGSConfig(prune_opacity_early=0.005, vcp_subsample_frac=0.5, prune_score_thresh=0.9)
    n = 100
    # High VCP everywhere, but opacity is healthy — official densify prune should remove none.
    scores = torch.ones(n)
    opa = torch.logit(torch.full((n,), 0.1))
    log_scales = torch.full((n, 3), math.log(0.02))
    prune = prune_mask_fastgs_densify(
        opa, log_scales, scores, scene_scale=1.0, cfg=cfg, size_threshold=None
    )
    assert int(prune.sum().item()) == 0


def test_densify_prune_low_opacity_with_budget():
    cfg = FastGSConfig(prune_opacity_early=0.005, vcp_subsample_frac=0.5)
    opa = torch.logit(torch.tensor([0.001, 0.001, 0.001, 0.1]))
    log_scales = torch.full((4, 3), math.log(0.02))
    scores = torch.zeros(4)
    prune = prune_mask_fastgs_densify(
        opa, log_scales, scores, scene_scale=1.0, cfg=cfg, size_threshold=None
    )
    # Official samples budget=1 from scored Gaussians, then ANDs with the low-opa set.
    # The healthy Gaussian is never pruned; at most one low-opa row is removed.
    assert int(prune.sum().item()) <= 1
    assert prune[-1].item() is False


def test_densify_prune_screen_and_scale_after_reset():
    cfg = FastGSConfig(prune_opacity_early=0.005, prune_scale3d=0.1)
    opa = torch.logit(torch.tensor([0.2, 0.2, 0.2]))
    log_scales = torch.log(torch.tensor([[0.01, 0.01, 0.01], [0.5, 0.5, 0.5], [0.01, 0.01, 0.01]]))
    radii = torch.tensor([1.0, 1.0, 40.0])
    scores = torch.zeros(3)
    prune = prune_mask_fastgs_densify(
        opa,
        log_scales,
        scores,
        scene_scale=1.0,
        cfg=cfg,
        max_radii2d=radii,
        size_threshold=20.0,
    )
    # 2 candidates (huge scale + 40px), budget = 1; official AND can drop the sample.
    assert int(prune.sum().item()) <= 1
    assert prune[0].item() is False


def test_densify_prune_does_not_prefer_unscored_new_gaussians():
    cfg = FastGSConfig(prune_opacity_early=0.005, vcp_subsample_frac=0.5)
    torch.manual_seed(0)
    # 2 originals with VCP scores, 2 clone/split children with no score yet.
    opa = torch.logit(torch.full((4,), 0.001))
    log_scales = torch.full((4, 3), math.log(0.02))
    scores = torch.tensor([0.9, 0.1])
    prune = prune_mask_fastgs_densify(
        opa, log_scales, scores, scene_scale=1.0, cfg=cfg, size_threshold=None
    )
    assert prune[2].item() is False
    assert prune[3].item() is False
    assert int(prune[:2].sum().item()) == 2


def test_late_vcp_uses_opacity_and_score_threshold():
    cfg = FastGSConfig(densify_until=15000, prune_opacity_late=0.1, prune_score_thresh=0.9)
    scores = torch.tensor([0.95, 0.1, 0.1])
    opa = torch.logit(torch.tensor([0.5, 0.05, 0.5]))
    prune = prune_mask_vcp(scores, opa, step=18000, cfg=cfg)
    assert prune.tolist() == [True, True, False]
