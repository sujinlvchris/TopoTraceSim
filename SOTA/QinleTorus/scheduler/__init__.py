"""QinleTorus scheduler package.

Modules:
    topology   --- ND torus coordinate / distance / path expansion
    timing     --- analytical per-hop cycle estimator
    direct     --- direct A2A comparison: one packet per (src, dst), full message
    dimrotation--- DimRotation: each chunk split into per-dim hops
"""

from . import topology, timing, direct, dimrotation  # noqa: F401
