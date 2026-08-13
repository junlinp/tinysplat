"""FastGS multi-view consistent densification (VCD) and pruning (VCP).

Implements the scoring from arXiv:2511.04283 on top of TinySplat training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FastGSConfig:
    k_views: int = 10
    error_tau: float = 0.5  # threshold on min-max normalized L1 map
    densify_score_thresh: float = 5.0  # τ_d
    prune_score_thresh: float = 0.9  # τ_p after 15k
    densify_every: int = 500
    densify_until: int = 15000
    prune_every_early: int = 500
    prune_every_late: int = 3000
    prune_opacity_early: float = 0.005
    prune_opacity_late: float = 0.1
    vcp_subsample_frac: float = 0.5  # before 15k: prune this fraction of high-score candidates
    absgrad_split: bool = True
    grad_thresh: float = 0.0002
    grow_scale3d: float = 0.01
    split_scale_shrink: float = 0.8
    # 3DGS-accel optimizer skip schedule
    skip_start: int = 15000
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
    mse = F.mse_loss(rendered, target)  # use L1 for FastGS Eq.10
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
    """Average high-error footprint counts across K views. Shape [N]."""
    stacked = torch.stack([c.float() for c in counts_per_view], dim=0)
    return stacked.mean(dim=0)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (clone_mask, split_mask) using VCD + gradient rules."""
    high = scores > cfg.densify_score_thresh
    if cfg.absgrad_split:
        grad_ok = grad_norms > cfg.grad_thresh
    else:
        grad_ok = grad_norms > cfg.grad_thresh
    cand = high & grad_ok
    max_scale = torch.exp(log_scales).max(dim=-1).values
    is_small = max_scale <= cfg.grow_scale3d * scene_scale
    clone_mask = cand & is_small
    split_mask = cand & ~is_small
    return clone_mask, split_mask


def prune_mask_vcp(
    scores: torch.Tensor,
    opacities_logit: torch.Tensor,
    step: int,
    cfg: FastGSConfig,
) -> torch.Tensor:
    """Return boolean prune mask."""
    opa = torch.sigmoid(opacities_logit)
    if step <= cfg.densify_until:
        low_opa = opa < cfg.prune_opacity_early
        # Subsample high VCP-score candidates.
        high = scores > 0.5
        n_high = int(high.sum().item())
        keep_high = torch.zeros_like(high)
        if n_high > 0:
            idx = torch.where(high)[0]
            k = max(1, int(n_high * cfg.vcp_subsample_frac))
            # Highest scores pruned preferentially.
            order = torch.argsort(scores[idx], descending=True)
            chosen = idx[order[:k]]
            keep_high[chosen] = True
        return low_opa | keep_high
    # Late phase
    return (opa < cfg.prune_opacity_late) | (scores > cfg.prune_score_thresh)
