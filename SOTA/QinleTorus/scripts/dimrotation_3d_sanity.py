#!/usr/bin/env python3
"""Minimal 3D Torus DimRotation reproduction.

This script reproduces only the chunk-to-dimension rotation mechanism:

1. Split each node's data into D chunks, where D is the torus dimension count.
2. Let chunk k start from dimension k.
3. Let every chunk visit all dimensions in cyclic order.

For a 3D Torus this gives:

    chunk 1: X -> Y -> Z
    chunk 2: Y -> Z -> X
    chunk 3: Z -> X -> Y

The assertions below make the script usable as a small sanity test.
"""

from __future__ import annotations

from collections.abc import Iterable


AXES = ("X", "Y", "Z")


def dimRotation(chunkId: int, dims: int) -> tuple[int, ...]:
    """Return the cyclic dimension order assigned to one chunk."""
    if dims <= 0:
        raise ValueError("dims must be positive")
    return tuple((chunkId + offset) % dims for offset in range(dims))


def expandDimOrderPath(
    src: tuple[int, ...],
    dst: tuple[int, ...],
    dimOrder: Iterable[int],
) -> list[tuple[int, ...]]:
    """Expand one chunk route into D single-dimension hops."""
    current = list(src)
    path = [src]

    for dim in dimOrder:
        current[dim] = dst[dim]
        path.append(tuple(current))

    return path


def changedDimension(src: tuple[int, ...], dst: tuple[int, ...]) -> int:
    """Return the only dimension changed by one DimRotation hop."""
    changed = [dim for dim, (srcCoord, dstCoord) in enumerate(zip(src, dst)) if srcCoord != dstCoord]

    if len(changed) != 1:
        raise AssertionError(f"expected exactly one changed dimension, got {changed}")

    return changed[0]


def formatOrder(dimOrder: Iterable[int]) -> str:
    """Format a dimension order as X->Y->Z style text."""
    return "->".join(AXES[dim] for dim in dimOrder)


def main() -> None:
    dims = 3
    chunks = dims
    src = (0, 0, 0)
    dst = (1, 2, 3)

    expectedOrders = (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    )

    firstPhaseDims: list[int] = []

    print("3D Torus DimRotation minimal reproduction")
    print(f"src={src}, dst={dst}, dims={dims}, chunks={chunks}")

    for chunkId in range(chunks):
        dimOrder = dimRotation(chunkId, dims)
        path = expandDimOrderPath(src, dst, dimOrder)

        assert dimOrder == expectedOrders[chunkId]

        print(f"chunk {chunkId + 1}: {formatOrder(dimOrder)}")

        observedDims: list[int] = []
        for phase, (hopSrc, hopDst) in enumerate(zip(path, path[1:]), start=1):
            dim = changedDimension(hopSrc, hopDst)
            observedDims.append(dim)
            print(f"  phase {phase}: {hopSrc} -> {hopDst}  dim={AXES[dim]}")

        assert tuple(observedDims) == dimOrder
        firstPhaseDims.append(observedDims[0])

    assert set(firstPhaseDims) == set(range(dims))
    print("PASS: first phase covers X, Y, and Z simultaneously.")


if __name__ == "__main__":
    main()
