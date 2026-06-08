"""Analytical per-hop cycle estimator.

Constants mirror popnet_anytopo/index.h:
    PIPE_DELAY_  = 1.0   (per router stage)
    WIRE_DELAY_  = 0.9   (per inter-router link)

A single hop in dim d covering ``h`` torus hops with ``flits`` flits is
approximated as:

    head_latency = h * (PIPE_DELAY + WIRE_DELAY)
    body_serialize = flits     # wormhole, 1 flit / cycle on the link
    cycles = head_latency + body_serialize
    cycles_with_slack = cycles * (1 + slack)

The estimate's only purpose is to schedule the NEXT hop's inject time so the
arriving flits don't queue up unnecessarily.  PopNet still simulates real
contention; the schedule is only a hint.
"""

from __future__ import annotations

from dataclasses import dataclass

from .topology import TorusGeom


PIPE_DELAY = 1.0
WIRE_DELAY = 0.9


@dataclass(frozen=True)
class TimingParams:
    pipe_delay: float = PIPE_DELAY
    wire_delay: float = WIRE_DELAY
    slack: float = 0.10


def estimate_hop_cycles(
    geom: TorusGeom,
    src: tuple[int, ...],
    dst: tuple[int, ...],
    flits: int,
    params: TimingParams = TimingParams(),
) -> int:
    """Estimate cycles for one single-dim hop (src and dst differ in <=1 coord)."""
    diff_dims = [i for i, (a, b) in enumerate(zip(src, dst)) if a != b]
    if len(diff_dims) == 0:
        return 0
    if len(diff_dims) > 1:
        raise ValueError(
            f"estimate_hop_cycles expects single-dim hop, got src={src} dst={dst}"
        )
    d = diff_dims[0]
    h = geom.torus_hops_in_dim(src[d], dst[d])
    head = h * (params.pipe_delay + params.wire_delay)
    serial = float(flits)
    cycles = (head + serial) * (1.0 + params.slack)
    return int(cycles) + 1  # +1 keeps cycle strictly larger than zero


def estimate_packet_cycles(
    geom: TorusGeom,
    src: tuple[int, ...],
    dst: tuple[int, ...],
    flits: int,
    params: TimingParams = TimingParams(),
) -> int:
    """Estimate cycles for a single popnet packet that may span multiple dims
    (used by direct A2A comparison mode).  Uses Manhattan torus distance."""
    h = geom.torus_distance(src, dst)
    head = h * (params.pipe_delay + params.wire_delay)
    serial = float(flits)
    cycles = (head + serial) * (1.0 + params.slack)
    return int(cycles) + 1
