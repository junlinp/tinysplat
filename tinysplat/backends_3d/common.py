"""Shared backend types for 3D TinySplat renderers."""

from dataclasses import dataclass
from typing import Callable

import torch


Render3DFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Backend3DOps:
    """One 3D renderer backend implementation."""

    name: str
    render: Render3DFn
    is_compiled: bool = False
