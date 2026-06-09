#!/usr/bin/env python3
"""Emit per-token tile routes from a real Switch HF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_switch_hf_moe_layer import load_checkpoint_state, load_hidden_states
from run_switch_moe_layer import import_pytorchsim


def tile_coord(tile_id: int, tile_cols: int) -> tuple[int, int]:
    return tile_id // tile_cols, tile_id % tile_cols


def chiplet_coord(chiplet_id: int, chiplet_cols: int) -> tuple[int, int]:
    return chiplet_id // chiplet_cols, chiplet_id % chiplet_cols


def global_tile(chiplet_id: int, tile_id: int, chiplet_cols: int, tile_rows: int, tile_cols: int) -> tuple[int, int]:
    chiplet_r, chiplet_c = chiplet_coord(chiplet_id, chiplet_cols)
    local_r, local_c = tile_coord(tile_id, tile_cols)
    return chiplet_r * tile_rows + local_r, chiplet_c * tile_cols + local_c


def build_token_routes(
    assignments: Any,
    nodes: int,
    tokens_per_source: int,
    num_experts: int,
    expert_capacity: int,
    chiplet_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> dict[str, Any]:
    experts_per_chiplet = num_experts // nodes
    tiles_per_chiplet = tile_rows * tile_cols
    expert_fill = [0 for _ in range(num_experts)]
    raw_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    kept_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    dropped_counts = [[0 for _ in range(num_experts)] for _ in range(nodes)]
    token_routes: list[dict[str, Any]] = []

    for token_id in range(tokens_per_source):
        for src_chiplet in range(nodes):
            expert_id = int(assignments[src_chiplet, token_id])
            dst_chiplet = expert_id // experts_per_chiplet
            raw_counts[src_chiplet][expert_id] += 1

            src_tile_id = token_id % tiles_per_chiplet
            src_global_r, src_global_c = global_tile(
                src_chiplet,
                src_tile_id,
                chiplet_cols,
                tile_rows,
                tile_cols,
            )

            kept = expert_capacity <= 0 or expert_fill[expert_id] < expert_capacity
            dst_tile_id = None
            dst_global_r = None
            dst_global_c = None
            if kept:
                dst_tile_id = expert_fill[expert_id] % tiles_per_chiplet
                dst_global_r, dst_global_c = global_tile(
                    dst_chiplet,
                    dst_tile_id,
                    chiplet_cols,
                    tile_rows,
                    tile_cols,
                )
                expert_fill[expert_id] += 1
                kept_counts[src_chiplet][expert_id] += 1
            else:
                dropped_counts[src_chiplet][expert_id] += 1

            token_routes.append({
                "token_id": token_id,
                "src_chiplet": src_chiplet,
                "src_tile_id": src_tile_id,
                "src_global_row": src_global_r,
                "src_global_col": src_global_c,
                "expert_id": expert_id,
                "dst_chiplet": dst_chiplet,
                "dst_tile_id": dst_tile_id,
                "dst_global_row": dst_global_r,
                "dst_global_col": dst_global_c,
                "is_intra_chiplet": src_chiplet == dst_chiplet,
                "kept": kept,
            })

    return {
        "routing_source": "real_huggingface_checkpoint_hidden_states_per_token",
        "nodes": nodes,
        "tokens_per_source": tokens_per_source,
        "num_experts": num_experts,
        "expert_capacity": expert_capacity,
        "experts_per_chiplet": experts_per_chiplet,
        "tile_rows_per_chiplet": tile_rows,
        "tile_cols_per_chiplet": tile_cols,
        "tiles_per_chiplet": tiles_per_chiplet,
        "raw_counts_by_source_expert": raw_counts,
        "kept_counts_by_source_expert": kept_counts,
        "dropped_counts_by_source_expert": dropped_counts,
        "kept_counts_by_expert": expert_fill,
        "total_tokens": nodes * tokens_per_source,
        "total_kept_tokens": sum(expert_fill),
        "total_dropped_tokens": sum(sum(row) for row in dropped_counts),
        "token_routes": token_routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real Switch per-token tile routes.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint-file", default="pytorch_model.bin")
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--router-key", default="encoder.block.1.layer.1.mlp.router.classifier.weight")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-capacity", type=int, default=64)
    parser.add_argument("--chiplet-cols", type=int, default=2)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nodes = args.ary ** args.dims
    torch, _device, _tog_simulator_cls, _sim_backend = import_pytorchsim()
    state = load_checkpoint_state(torch, Path(args.model_dir), args.checkpoint_file)
    router_weight = state[args.router_key].float().contiguous()
    router_weight_t = router_weight.t().contiguous()
    hidden, hidden_meta = load_hidden_states(
        torch,
        Path(args.hidden_states),
        nodes=nodes,
        tokens_per_source=args.tokens_per_source,
        d_model=args.d_model,
    )

    logits = torch.matmul(hidden, router_weight_t)
    assignments = torch.argmax(logits, dim=-1)
    payload = build_token_routes(
        assignments=assignments,
        nodes=nodes,
        tokens_per_source=args.tokens_per_source,
        num_experts=args.num_experts,
        expert_capacity=args.expert_capacity,
        chiplet_cols=args.chiplet_cols,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    )
    payload.update({
        "model_dir": str(Path(args.model_dir)),
        "hidden_states": str(Path(args.hidden_states)),
        "router_key": args.router_key,
        "hidden_meta": {
            key: value
            for key, value in hidden_meta.items()
            if key not in {"input_ids", "attention_mask", "texts"}
        },
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote tile routes -> {out_path}")
    print(f"raw_counts_by_source_expert={payload['raw_counts_by_source_expert']}")
    print(f"kept_counts_by_source_expert={payload['kept_counts_by_source_expert']}")
    print(f"total_kept_tokens={payload['total_kept_tokens']}")
    print(f"total_dropped_tokens={payload['total_dropped_tokens']}")


if __name__ == "__main__":
    main()
