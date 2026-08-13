"""Root helpers (Metal / FastGS / SH). Prefer `pip install -e legacy` for training APIs."""

from __future__ import annotations

__version__ = "0.2.0"

try:
    from .metal_backend import (  # noqa: F401
        count_footprint_hits,
        forward_3d,
        metal_available,
        render_metal_3d,
    )
except Exception:
    pass

try:
    from .fastgs import FastGSConfig  # noqa: F401
    from .sh import colors_from_sh, init_sh_from_rgb, sh_degree_from_step  # noqa: F401
except Exception:
    pass
