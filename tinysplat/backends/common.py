"""Shared backend types for TinySplat."""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


ForwardFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    Tuple[torch.Tensor, list],
]
BackwardFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        list,
        tuple,
    ],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class BackendOps:
    """Forward and backward callables for one backend."""

    name: str
    forward: ForwardFn
    backward: BackwardFn
    is_compiled: bool = False
