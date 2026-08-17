"""FastGS multi-view consistent densification (VCD) and pruning (VCP).

Implements the scoring from arXiv:2511.04283 on top of TinySplat training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FastGSConfig:
    k_views: int = 10
    error_tau: float = 0.1  # official FastGS loss_thresh on min-max L1
    densify_score_thresh: float = 5.0  # τ_d
    prune_score_thresh: float = 0.9  # τ_p (late prune only)
    densify_every: int = 500
    densify_until: int = 15000
    prune_every_early: int = 500
    prune_every_late: int = 3000
    prune_opacity_early: float = 0.005
    prune_opacity_late: float = 0.1
    prune_scale3d: float = 0.1
    max_screen_size: float = 20.0  # pixels; official size_threshold after opacity reset
    vcp_subsample_frac: float = 0.5  # densify-phase prune budget vs low-opa/big set
    absgrad_split: bool = True
    grad_thresh: float = 0.0002
    grad_abs_thresh: float = 0.0009  # truck FastGS-base; code default is 0.0012
    grow_scale3d: float = 0.001  # official percent_dense
    split_scale_shrink: float = 1.6
    cull_screen_size: float = 0.0  # FastGS does not use 15% screen cull
    # 3DGS-accel optimizer skip schedule
    skip_start: int = 1_000_000_000  # train_base uses every-step Adam; 3DGS-accel skips after 15k
    skip_mid: int = 20000
    skip_every_mid: int = 32
    skip_every_late: int = 64
    ssim_lambda: float = 0.2


def should_step_optimizer(step: int, cfg: FastGSConfig) -> bool:
    """3DGS-accel style optimizer skip after densify phase."""
    if step < cfg.skip_start:
        return True
    if step < cfg.skip_mid:
        return (step % cfg.skip_every_mid) == 0
    return (step % cfg.skip_every_late) == 0


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo + 1e-8)


def screen_frac_from_proj_covs(
    proj_covs: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Largest 3-sigma ellipse axis as a fraction of max(H, W). Shape [N]."""
    cov_xx = proj_covs[:, 0, 0]
    cov_xy = proj_covs[:, 0, 1]
    cov_yy = proj_covs[:, 1, 1]
    trace = cov_xx + cov_yy
    disc = torch.sqrt(torch.clamp((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, min=0.0))
    lambda_max = 0.5 * (trace + disc)
    screen_radius = 3.0 * torch.sqrt(torch.clamp(lambda_max, min=0.0))
    return screen_radius / float(max(int(height), int(width), 1))


def split_child_log_scales(log_scales: torch.Tensor, shrink: float) -> torch.Tensor:
    """Log-scales for split children.

    ``shrink < 1`` multiplies (nerfstudio / this repo default 0.8).
    ``shrink >= 1`` divides (vanilla 3DGS uses 1.6).
    """
    scale = float(shrink)
    if scale <= 0:
        scale = 0.8
    delta = math.log(scale) if scale < 1.0 else -math.log(scale)
    return log_scales + delta


@torch.no_grad()
def high_error_mask(
    rendered: torch.Tensor,
    target: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Per-pixel L1 mean over channels, min-max normalized, thresholded. Returns HxW uint8."""
    err = (rendered - target).abs().mean(dim=-1)
    m = minmax_norm(err)
    return (m > tau).to(torch.uint8)


@torch.no_grad()
def photometric_loss_value(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_fn,
    ssim_lambda: float,
) -> float:
    l1 = F.l1_loss(rendered, target)
    if ssim_fn is None or ssim_lambda <= 0:
        return float(l1.item())
    ssim = ssim_fn(
        rendered.permute(2, 0, 1).unsqueeze(0),
        target.permute(2, 0, 1).unsqueeze(0),
        data_range=1.0,
        size_average=True,
    )
    return float(((1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - ssim)).item())


@torch.no_grad()
def accumulate_vcd_scores(
    counts_per_view: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Floor-mean high-error compositor counts across K views. Shape [N].

    Official FastGS: ``floor(sum_j counts_j / K)``.
    """
    stacked = torch.stack([c.float() for c in counts_per_view], dim=0)
    return torch.div(stacked.sum(dim=0), stacked.shape[0], rounding_mode="floor")


@torch.no_grad()
def accumulate_vcp_scores(
    counts_per_view: Sequence[torch.Tensor],
    photo_per_view: Sequence[float],
) -> torch.Tensor:
    """Normalize sum_j (hits_j * E_photo_j). Shape [N]."""
    acc = torch.zeros_like(counts_per_view[0], dtype=torch.float32)
    for counts, e in zip(counts_per_view, photo_per_view):
        acc = acc + counts.float() * float(e)
    return minmax_norm(acc)


def densify_mask_vcd(
    scores: torch.Tensor,
    grad_norms: torch.Tensor,
    log_scales: torch.Tensor,
    cfg: FastGSConfig,
    scene_scale: float = 1.0,
    abs_grad_norms: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (clone_mask, split_mask) using VCD + gradient rules.

    FastGS clones on vanilla (signed) grad and splits on AbsGS when
    ``abs_grad_norms`` is provided and ``cfg.absgrad_split`` is set.
    """
    high = scores > cfg.densify_score_thresh
    clone_grad_ok = grad_norms >= cfg.grad_thresh
    split_grad = (
        abs_grad_norms
        if cfg.absgrad_split and abs_grad_norms is not None
        else grad_norms
    )
    split_thresh = (
        cfg.grad_abs_thresh
        if cfg.absgrad_split and abs_grad_norms is not None
        else cfg.grad_thresh
    )
    split_grad_ok = split_grad >= split_thresh
    max_scale = torch.exp(log_scales).max(dim=-1).values
    is_small = max_scale <= cfg.grow_scale3d * scene_scale
    clone_mask = high & clone_grad_ok & is_small
    split_mask = high & split_grad_ok & ~is_small
    return clone_mask, split_mask


def prune_mask_vcp(
    scores: torch.Tensor,
    opacities_logit: torch.Tensor,
    step: int,
    cfg: FastGSConfig,
    log_scales: Optional[torch.Tensor] = None,
    scene_scale: float = 1.0,
) -> torch.Tensor:
    """Return boolean prune mask.

    During densify (step <= densify_until): official FastGS densify prune (low
    opacity, 50% VCP-weighted subsample). After that: opacity < 0.1 or VCP > τ_p.
    """
    opa = torch.sigmoid(opacities_logit).reshape(-1)
    if step <= cfg.densify_until:
        return prune_mask_fastgs_densify(
            opacities_logit,
            log_scales,
            scores,
            scene_scale,
            cfg,
            max_radii2d=None,
            size_threshold=None,
        )
    return (opa < cfg.prune_opacity_late) | (scores.reshape(-1) > cfg.prune_score_thresh)


def prune_mask_fastgs_densify(
    opacities_logit: torch.Tensor,
    log_scales: Optional[torch.Tensor],
    pruning_score: torch.Tensor,
    scene_scale: float,
    cfg: FastGSConfig,
    max_radii2d: Optional[torch.Tensor] = None,
    size_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Official densify-phase prune: low opacity / oversized, then 50% VCP-weighted sample.

    FastGS does **not** independently prune on VCP > τ_p during densification. VCP only
    weights which of the low-opacity / large Gaussians are removed (budget = half).
    """
    opa = torch.sigmoid(opacities_logit).reshape(-1)
    prune = opa < cfg.prune_opacity_early
    if size_threshold is not None:
        if max_radii2d is not None and max_radii2d.numel() == prune.numel():
            prune = prune | (max_radii2d.reshape(-1) > float(size_threshold))
        if log_scales is not None:
            max_scale = torch.exp(log_scales).max(dim=-1).values
            prune = prune | (max_scale > cfg.prune_scale3d * scene_scale)

    n = prune.numel()
    cand_idx = torch.nonzero(prune, as_tuple=False).reshape(-1)
    to_remove = int(cand_idx.numel())
    budget = int(cfg.vcp_subsample_frac * to_remove)
    if budget <= 0:
        return torch.zeros_like(prune)

    # Official pads multinomial weights with 0 for Gaussians that have no VCP
    # yet (clone/split children). Weighting those as 1/(1e-6+0) preferentially
    # deletes the new points we just added.
    scores = torch.zeros(n, device=prune.device, dtype=torch.float32)
    ps = pruning_score.reshape(-1).to(dtype=torch.float32)
    n_score = min(int(ps.numel()), n)
    if n_score > 0:
        scores[:n_score] = 1.0 - ps[:n_score]
    weights = torch.zeros(n, device=prune.device, dtype=torch.float32)
    if n_score > 0:
        weights[:n_score] = 1.0 / (1e-6 + scores[:n_score])
    weights = torch.clamp(weights, min=0.0)

    # Sample the budget *among the prune candidates only*. Drawing from the full
    # population and intersecting afterwards throws most of the budget away: with
    # 8k candidates in 100k Gaussians it removed ~8% of the intended count, so
    # low-opacity Gaussians accumulated instead of being cleaned up.
    cand_weights = weights[cand_idx]
    nonzero = int((cand_weights > 0).sum().item())
    if nonzero <= 0:
        return torch.zeros_like(prune)
    # multinomial without replacement only avoids zero-weight entries while the
    # draw count stays within their number; clamp so unscored children survive.
    num_samples = min(budget, nonzero)
    sampled_local = torch.multinomial(cand_weights, num_samples=num_samples, replacement=False)
    out = torch.zeros_like(prune)
    out[cand_idx[sampled_local]] = True
    return out
