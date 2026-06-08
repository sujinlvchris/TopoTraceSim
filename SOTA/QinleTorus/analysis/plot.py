#!/usr/bin/env python3
"""Render direct-A2A vs DimRotation comparison plots from sweep JSON files.

Reads every ``*.json`` under ``--results-dir`` (skipping non-metric files),
groups them by (torus, msg_size), and emits:

* ``a2a_latency.png``        bar chart of A2A latency cycles
* ``dim_balance.png``        per-dim event distribution
* ``throughput.png``         flits / cycle

Usage::

    python analysis/plot.py --results-dir results/ --out-dir results/figures/
"""

from __future__ import annotations

import argparse
import struct
import json
import zlib
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


def load_results(results_dir: Path):
    runs = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name.startswith("sweep_"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            print(f"skip {p}: {exc}")
            continue
        if "metrics" not in data:
            continue
        runs.append(data)
    return runs


def key_for(run):
    g = run["geom"]
    bench_path = Path(run["bench"]["path"]).name
    flits = run["bench"]["total_flits"]
    return (f"{g['ary']}x{g['ary']}", flits, bench_path)


def grouped(runs):
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        g = r["geom"]
        torus = f"{g['ary']}x{g['ary']}"
        bench_name = Path(r["bench"]["path"]).parent.name  # e.g. torus_4x4_16KB_direct
        # take "<torus>_<msg>" prefix as group
        prefix = "_".join(bench_name.split("_")[:-1]) or bench_name
        groups.setdefault((torus, prefix), []).append(r)
    return groups


def bar_compare(groups, key_path, ylabel, title, out_path):
    if not HAS_MATPLOTLIB:
        simple_bar_png(groups, key_path, title, out_path)
        return
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.3), 4.0))
    labels = []
    direct_vals = []
    dimrot_vals = []
    for (torus, prefix), runs in sorted(groups.items()):
        d_run = next((r for r in runs if r["label"] == "direct"), None)
        r_run = next((r for r in runs if r["label"] == "dimrotation"), None)
        labels.append(prefix)
        direct_vals.append(_dig(d_run, key_path))
        dimrot_vals.append(_dig(r_run, key_path))

    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], direct_vals, width=width, label="direct")
    ax.bar([i + width / 2 for i in x], dimrot_vals, width=width, label="dimrotation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_dim_stack(groups, dims, out_path):
    if not HAS_MATPLOTLIB:
        # Fallback: total per-dim events grouped as direct/dimrotation bars.
        simple_bar_png(groups, ["metrics", "total_wire_events"],
                       "Total wire events", out_path)
        return
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 4.5))
    labels = []
    direct_per_dim = [[] for _ in range(dims)]
    dimrot_per_dim = [[] for _ in range(dims)]
    for (torus, prefix), runs in sorted(groups.items()):
        d_run = next((r for r in runs if r["label"] == "direct"), None)
        r_run = next((r for r in runs if r["label"] == "dimrotation"), None)
        labels.append(prefix + "\ndirect")
        labels.append(prefix + "\ndimrot")
        for dim in range(dims):
            direct_per_dim[dim].append(_dig(d_run, ["metrics", "per_dim_events", str(dim)]))
            dimrot_per_dim[dim].append(_dig(r_run, ["metrics", "per_dim_events", str(dim)]))

    interleaved = []
    for i in range(len(direct_per_dim[0])):
        for dim in range(dims):
            interleaved.append((direct_per_dim[dim][i], dimrot_per_dim[dim][i]))

    x = range(len(labels))
    bottoms = [0] * len(labels)
    for dim in range(dims):
        vals = []
        for i in range(len(direct_per_dim[0])):
            vals.append(direct_per_dim[dim][i])
            vals.append(dimrot_per_dim[dim][i])
        ax.bar(list(x), vals, bottom=bottoms, label=f"dim {dim}")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("wire events")
    ax.set_title("Per-dim wire-event load")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _dig(run, path):
    if not run:
        return 0
    cur = run
    for p in path:
        if cur is None:
            return 0
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return 0
    if isinstance(cur, (int, float)):
        return cur
    return 0


def write_summary(groups, out_path):
    rows = []
    rows.append("torus,config,scheduler,a2a_latency,total_flits,payload_bytes,throughput_fpc,throughput_GBps,dim_balance,total_events")
    for (torus, prefix), runs in sorted(groups.items()):
        for r in sorted(runs, key=lambda r: r["label"]):
            m = r["metrics"]
            rows.append(
                ",".join(str(x) for x in [
                    torus,
                    prefix,
                    r["label"],
                    m["a2a_latency_cycles"],
                    r["bench"]["total_flits"],
                    r["bench"].get("payload_bytes", m.get("total_bytes", "")),
                    m["throughput_flits_per_cycle"],
                    m.get("throughput_GBps", ""),
                    m["dim_load_balance"],
                    m["total_wire_events"],
                ])
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")


def simple_bar_png(groups, key_path, title, out_path):
    """Tiny stdlib PNG fallback for servers without matplotlib."""
    labels = []
    direct_vals = []
    dimrot_vals = []
    for (_torus, prefix), runs in sorted(groups.items()):
        d_run = next((r for r in runs if r["label"] == "direct"), None)
        r_run = next((r for r in runs if r["label"] == "dimrotation"), None)
        labels.append(prefix)
        direct_vals.append(float(_dig(d_run, key_path)))
        dimrot_vals.append(float(_dig(r_run, key_path)))
    max_v = max(direct_vals + dimrot_vals + [1.0])
    width = max(640, 130 * max(1, len(labels)))
    height = 420
    img = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    def rect(x0, y0, x1, y1, color):
        x0, x1 = max(0, int(x0)), min(width, int(x1))
        y0, y1 = max(0, int(y0)), min(height, int(y1))
        for y in range(y0, y1):
            row = img[y]
            for x in range(x0, x1):
                row[x] = color

    plot_left, plot_top, plot_right, plot_bottom = 60, 40, width - 20, height - 80
    rect(plot_left, plot_bottom, plot_right, plot_bottom + 2, (0, 0, 0))
    rect(plot_left, plot_top, plot_left + 2, plot_bottom, (0, 0, 0))
    group_w = (plot_right - plot_left) / max(1, len(labels))
    bar_w = min(24, group_w * 0.28)
    for i, (dv, rv) in enumerate(zip(direct_vals, dimrot_vals)):
        center = plot_left + group_w * (i + 0.5)
        for val, dx, color in [
            (dv, -bar_w * 0.65, (72, 118, 255)),
            (rv, bar_w * 0.65, (255, 126, 64)),
        ]:
            h = (plot_bottom - plot_top) * (val / max_v)
            rect(center + dx - bar_w / 2, plot_bottom - h, center + dx + bar_w / 2, plot_bottom, color)

    # Minimal legend blocks (no text; CSV has exact labels/values).
    rect(width - 130, 20, width - 112, 38, (72, 118, 255))
    rect(width - 130, 45, width - 112, 63, (255, 126, 64))
    write_png(out_path, img)


def write_png(path: Path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--dims", type=int, default=2)
    args = ap.parse_args()

    runs = load_results(Path(args.results_dir))
    if not runs:
        print(f"no metric JSONs found under {args.results_dir}")
        return

    groups = grouped(runs)
    out_dir = Path(args.out_dir)

    bar_compare(groups, ["metrics", "a2a_latency_cycles"],
                "A2A latency (cycles)", "A2A latency: direct A2A vs DimRotation",
                out_dir / "a2a_latency.png")
    bar_compare(groups, ["metrics", "throughput_GBps"],
                "GB/s", "Throughput: direct A2A vs DimRotation",
                out_dir / "throughput.png")
    bar_compare(groups, ["metrics", "dim_load_balance"],
                "balance (1=perfect)", "Per-dim load balance",
                out_dir / "dim_balance.png")
    per_dim_stack(groups, args.dims, out_dir / "per_dim_stack.png")
    write_summary(groups, out_dir / "summary.csv")

    print(f"wrote figures to {out_dir}")
    if not HAS_MATPLOTLIB:
        print("matplotlib not found; wrote simple stdlib PNGs plus summary.csv")


if __name__ == "__main__":
    main()
