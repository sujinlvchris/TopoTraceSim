#!/usr/bin/env python3
"""Plot tile-level Switch MoE load heatmaps with chiplet boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def chiplet_origin(chiplet_id: int, chiplet_cols: int, tile_rows: int, tile_cols: int) -> tuple[int, int]:
    chiplet_row = chiplet_id // chiplet_cols
    chiplet_col = chiplet_id % chiplet_cols
    return chiplet_row * tile_rows, chiplet_col * tile_cols


def parse_tile(spec: str) -> tuple[int, int]:
    row, col = spec.split(",")
    return int(row), int(col)


def router_global_tile(
    chiplet_id: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    router_local_tile: tuple[int, int],
) -> tuple[int, int]:
    base_r, base_c = chiplet_origin(chiplet_id, chiplet_cols, tile_rows, tile_cols)
    return base_r + router_local_tile[0], base_c + router_local_tile[1]


def xy_path(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    """Tile-level XY path from src to dst, including both endpoints."""
    sr, sc = src
    dr, dc = dst
    path = [(sr, sc)]

    step_c = 1 if dc >= sc else -1
    for col in range(sc + step_c, dc + step_c, step_c):
        path.append((sr, col))

    step_r = 1 if dr >= sr else -1
    for row in range(sr + step_r, dr + step_r, step_r):
        path.append((row, dc))

    return path


def add_path(matrix: np.ndarray, path: list[tuple[int, int]], weight: float) -> None:
    for row, col in path:
        matrix[row, col] += weight


def build_matrices_from_counts(
    routes: dict,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    intra_chiplet_weight: float,
    inter_chiplet_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = routes["kept_counts_by_source_expert"]
    nodes = int(routes["nodes"])
    num_experts = int(routes["num_experts"])
    experts_per_chiplet = num_experts // nodes

    height = chiplet_rows * tile_rows
    width = chiplet_cols * tile_cols
    intra = np.zeros((height, width), dtype=float)
    inter = np.zeros((height, width), dtype=float)

    for src_chiplet, row in enumerate(counts):
        for expert_id, tokens in enumerate(row):
            dst_chiplet = expert_id // experts_per_chiplet
            base_r, base_c = chiplet_origin(dst_chiplet, chiplet_cols, tile_rows, tile_cols)
            if src_chiplet == dst_chiplet:
                target = intra
                weight = intra_chiplet_weight
            else:
                target = inter
                weight = inter_chiplet_weight

            per_tile_tokens = float(tokens) * weight / float(tile_rows * tile_cols)
            target[base_r:base_r + tile_rows, base_c:base_c + tile_cols] += per_tile_tokens

    return intra + inter, intra, inter


def build_destination_matrices_from_token_routes(
    routes: dict,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    intra_chiplet_weight: float,
    inter_chiplet_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = chiplet_rows * tile_rows
    width = chiplet_cols * tile_cols
    intra = np.zeros((height, width), dtype=float)
    inter = np.zeros((height, width), dtype=float)

    for item in routes["token_routes"]:
        if not item["kept"]:
            continue
        row = int(item["dst_global_row"])
        col = int(item["dst_global_col"])
        if item["is_intra_chiplet"]:
            intra[row, col] += intra_chiplet_weight
        else:
            inter[row, col] += inter_chiplet_weight

    return intra + inter, intra, inter


def build_activity_matrices_from_token_routes(
    routes: dict,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    intra_chiplet_weight: float,
    inter_chiplet_weight: float,
    router_local_tile: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = chiplet_rows * tile_rows
    width = chiplet_cols * tile_cols
    intra = np.zeros((height, width), dtype=float)
    inter = np.zeros((height, width), dtype=float)

    for item in routes["token_routes"]:
        if not item["kept"]:
            continue

        src_chiplet = int(item["src_chiplet"])
        dst_chiplet = int(item["dst_chiplet"])
        src_tile = (int(item["src_global_row"]), int(item["src_global_col"]))
        dst_tile = (int(item["dst_global_row"]), int(item["dst_global_col"]))
        src_router = router_global_tile(src_chiplet, chiplet_cols, tile_rows, tile_cols, router_local_tile)

        if item["is_intra_chiplet"]:
            add_path(intra, xy_path(src_tile, src_router), intra_chiplet_weight)
            add_path(intra, xy_path(src_router, dst_tile), intra_chiplet_weight)
        else:
            dst_router = router_global_tile(dst_chiplet, chiplet_cols, tile_rows, tile_cols, router_local_tile)
            add_path(inter, xy_path(src_tile, src_router), intra_chiplet_weight)
            inter[src_router] += inter_chiplet_weight
            inter[dst_router] += inter_chiplet_weight
            add_path(inter, xy_path(dst_router, dst_tile), intra_chiplet_weight)

    return intra + inter, intra, inter


def build_expert_matrix_from_token_routes(
    routes: dict,
    chiplet_cols: int,
    intra_chiplet_weight: float,
    inter_chiplet_weight: float,
) -> np.ndarray:
    nodes = int(routes["nodes"])
    num_experts = int(routes["num_experts"])
    experts_per_chiplet = num_experts // nodes
    chiplet_rows = nodes // chiplet_cols
    matrix = np.zeros((chiplet_rows, chiplet_cols * experts_per_chiplet), dtype=float)

    for item in routes["token_routes"]:
        if not item["kept"]:
            continue
        dst_chiplet = int(item["dst_chiplet"])
        expert_id = int(item["expert_id"])
        local_expert = expert_id % experts_per_chiplet
        chiplet_row = dst_chiplet // chiplet_cols
        chiplet_col = dst_chiplet % chiplet_cols
        weight = intra_chiplet_weight if item["is_intra_chiplet"] else inter_chiplet_weight
        matrix[chiplet_row, chiplet_col * experts_per_chiplet + local_expert] += weight

    return matrix


def draw_chiplet_grid(ax, chiplet_rows: int, chiplet_cols: int, tile_rows: int, tile_cols: int) -> None:
    height = chiplet_rows * tile_rows
    width = chiplet_cols * tile_cols

    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for x in range(tile_cols, width, tile_cols):
        ax.axvline(x - 0.5, color="black", linewidth=2.2)
    for y in range(tile_rows, height, tile_rows):
        ax.axhline(y - 0.5, color="black", linewidth=2.2)

    for chiplet_id in range(chiplet_rows * chiplet_cols):
        base_r, base_c = chiplet_origin(chiplet_id, chiplet_cols, tile_rows, tile_cols)
        ax.text(
            base_c + 0.15,
            base_r + 0.25,
            f"C{chiplet_id}",
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        )


def annotate_nonzero(ax, data: np.ndarray, norm: colors.Normalize) -> None:
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if value <= 0:
                continue
            label = f"{value:.1f}" if abs(value - round(value)) > 1e-6 else f"{int(value)}"
            text_color = "black" if norm(value) > 0.58 else "white"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=text_color)


def fmt_value(value: float) -> str:
    return f"{value:.1f}" if abs(value - round(value)) > 1e-6 else f"{int(value)}"


def panel_norm(data: np.ndarray, gamma: float) -> colors.Normalize:
    positive = data[data > 0]
    if positive.size == 0:
        return colors.Normalize(vmin=0, vmax=1)

    vmin = float(positive.min())
    vmax = float(positive.max())
    if vmax <= vmin:
        return colors.Normalize(vmin=0, vmax=max(1.0, vmax))
    return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


def draw_expert_grid(
    ax,
    expert_matrix: np.ndarray,
    chiplet_cols: int,
    figure_unit: str,
    color_gamma: float,
):
    norm = panel_norm(expert_matrix, color_gamma)
    im = ax.imshow(expert_matrix, cmap="magma", norm=norm)
    rows, cols = expert_matrix.shape
    experts_per_chiplet = cols // chiplet_cols

    ax.set_title("Expert load", fontsize=12)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([f"slot {idx % experts_per_chiplet}" for idx in range(cols)])
    ax.set_yticklabels([f"chiplet row {idx}" for idx in range(rows)])
    ax.set_xlabel("expert slot inside each chiplet")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for x in range(experts_per_chiplet, cols, experts_per_chiplet):
        ax.axvline(x - 0.5, color="black", linewidth=2.2)
    if rows > 1:
        ax.axhline(0.5, color="black", linewidth=2.2)

    expert_id = 0
    for row in range(rows):
        for col in range(cols):
            value = expert_matrix[row, col]
            text_color = "black" if norm(value) > 0.62 else "white"
            ax.text(
                col,
                row,
                f"E{expert_id}\n{fmt_value(value)}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )
            expert_id += 1

    for chiplet_id in range(rows * chiplet_cols):
        chiplet_row = chiplet_id // chiplet_cols
        chiplet_col = chiplet_id % chiplet_cols
        ax.text(
            chiplet_col * experts_per_chiplet - 0.42,
            chiplet_row - 0.38,
            f"C{chiplet_id}",
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(figure_unit, fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def annotate_router_tiles(
    ax,
    data: np.ndarray,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    router_local_tile: tuple[int, int],
) -> None:
    for chiplet_id in range(chiplet_rows * chiplet_cols):
        row, col = router_global_tile(chiplet_id, chiplet_cols, tile_rows, tile_cols, router_local_tile)
        value = data[row, col]
        ax.text(
            col,
            row,
            f"R\n{fmt_value(value)}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65, "pad": 1.8},
        )


def draw_activity_grid(
    ax,
    data: np.ndarray,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    router_local_tile: tuple[int, int],
    figure_unit: str,
    color_gamma: float,
) -> None:
    im = ax.imshow(data, cmap="viridis", norm=panel_norm(data, color_gamma))
    ax.set_title("Tile path activity", fontsize=12)
    ax.set_xlabel("global tile col")
    ax.set_ylabel("global tile row")
    draw_chiplet_grid(ax, chiplet_rows, chiplet_cols, tile_rows, tile_cols)
    annotate_router_tiles(ax, data, chiplet_rows, chiplet_cols, tile_rows, tile_cols, router_local_tile)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(figure_unit, fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def plot_expert_activity(
    expert_matrix: np.ndarray,
    activity_matrix: np.ndarray,
    out_path: Path,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    router_local_tile: tuple[int, int],
    figure_title: str,
    unit_label: str,
    color_gamma: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )
    fig.suptitle(figure_title, fontsize=14)
    draw_expert_grid(axes[0], expert_matrix, chiplet_cols, unit_label, color_gamma)
    draw_activity_grid(
        axes[1],
        activity_matrix,
        chiplet_rows,
        chiplet_cols,
        tile_rows,
        tile_cols,
        router_local_tile,
        unit_label,
        color_gamma,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_heatmaps(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
    chiplet_rows: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
    figure_title: str,
    unit_label: str,
    color_gamma: float,
) -> None:
    titles = ["Total load", "Intra-chiplet load", "Inter-chiplet load"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), constrained_layout=True)
    fig.suptitle(figure_title, fontsize=14)
    for ax, data, title in zip(axes, matrices, titles):
        norm = panel_norm(data, color_gamma)
        im = ax.imshow(data, cmap="viridis", norm=norm)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("global tile col")
        ax.set_ylabel("global tile row")
        draw_chiplet_grid(ax, chiplet_rows, chiplet_cols, tile_rows, tile_cols)
        annotate_nonzero(ax, data, norm)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(unit_label, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2x2 chiplet / 4x4 tile Switch MoE heatmaps.")
    parser.add_argument("--routes-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matrix-out", default="")
    parser.add_argument("--chiplet-rows", type=int, default=2)
    parser.add_argument("--chiplet-cols", type=int, default=2)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=4)
    parser.add_argument("--intra-chiplet-weight", type=float, default=1.0)
    parser.add_argument("--inter-chiplet-weight", type=float, default=1.0)
    parser.add_argument("--layout", choices=["three-subfig", "expert-activity"], default="three-subfig")
    parser.add_argument("--load-model", choices=["activity", "destination"], default="activity")
    parser.add_argument("--router-local-tile", default="3,3")
    parser.add_argument("--figure-title", default="Switch MoE tile heatmap")
    parser.add_argument("--unit-label", default="tile activity")
    parser.add_argument("--color-gamma", type=float, default=0.75)
    args = parser.parse_args()

    routes = json.loads(Path(args.routes_json).read_text())
    if "token_routes" in routes:
        if args.load_model == "activity":
            matrices = build_activity_matrices_from_token_routes(
                routes=routes,
                chiplet_rows=args.chiplet_rows,
                chiplet_cols=args.chiplet_cols,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                intra_chiplet_weight=args.intra_chiplet_weight,
                inter_chiplet_weight=args.inter_chiplet_weight,
                router_local_tile=parse_tile(args.router_local_tile),
            )
        else:
            matrices = build_destination_matrices_from_token_routes(
                routes=routes,
                chiplet_rows=args.chiplet_rows,
                chiplet_cols=args.chiplet_cols,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                intra_chiplet_weight=args.intra_chiplet_weight,
                inter_chiplet_weight=args.inter_chiplet_weight,
            )
    else:
        matrices = build_matrices_from_counts(
            routes=routes,
            chiplet_rows=args.chiplet_rows,
            chiplet_cols=args.chiplet_cols,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            intra_chiplet_weight=args.intra_chiplet_weight,
            inter_chiplet_weight=args.inter_chiplet_weight,
        )

    expert_matrix = None
    if "token_routes" in routes:
        expert_matrix = build_expert_matrix_from_token_routes(
            routes=routes,
            chiplet_cols=args.chiplet_cols,
            intra_chiplet_weight=args.intra_chiplet_weight,
            inter_chiplet_weight=args.inter_chiplet_weight,
        )

    if args.layout == "expert-activity" and expert_matrix is not None:
        plot_expert_activity(
            expert_matrix=expert_matrix,
            activity_matrix=matrices[0],
            out_path=Path(args.out),
            chiplet_rows=args.chiplet_rows,
            chiplet_cols=args.chiplet_cols,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            router_local_tile=parse_tile(args.router_local_tile),
            figure_title=args.figure_title,
            unit_label=args.unit_label,
            color_gamma=args.color_gamma,
        )
    else:
        plot_heatmaps(
            matrices=matrices,
            out_path=Path(args.out),
            chiplet_rows=args.chiplet_rows,
            chiplet_cols=args.chiplet_cols,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            figure_title=args.figure_title,
            unit_label=args.unit_label,
            color_gamma=args.color_gamma,
        )

    if args.matrix_out:
        names = ["total", "intra", "inter"]
        payload = {
            "load_model": args.load_model,
            "router_local_tile": args.router_local_tile,
            "intra_chiplet_weight": args.intra_chiplet_weight,
            "inter_chiplet_weight": args.inter_chiplet_weight,
            "layout": args.layout,
            "expert_matrix": expert_matrix.tolist() if expert_matrix is not None else None,
            "matrices": {name: matrix.tolist() for name, matrix in zip(names, matrices)},
        }
        Path(args.matrix_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.matrix_out).write_text(json.dumps(payload, indent=2))

    print(f"wrote heatmap -> {args.out}")
    if args.matrix_out:
        print(f"wrote matrices -> {args.matrix_out}")


if __name__ == "__main__":
    main()
