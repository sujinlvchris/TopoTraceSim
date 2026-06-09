#!/usr/bin/env python3
"""Dump real Switch Transformer MoE router inputs from a local HF checkpoint.

This script runs the real HuggingFace ``google/switch-base-8`` encoder and
captures the tensor that enters a chosen sparse MoE router.  The generated
``.pt`` file is intentionally small and can be copied to the Linux server with
the checkpoint.  Server-side PyTorchSim then replays the same hidden states
through the real router weight to build a non-synthetic MoE A2A trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_TEXTS = [
    (
        "Mixture of Experts models route every token to a small subset of "
        "feed-forward experts. In a chiplet accelerator, the routing pattern "
        "creates all-to-all traffic because tokens produced on one compute "
        "chiplet may need to execute on experts mapped to other chiplets. "
        "The communication is naturally imbalanced when many tokens select "
        "the same expert while other experts remain lightly used. "
    ),
    (
        "Runtime reconfiguration changes the interposer-level connection "
        "pattern at communication phase boundaries. A useful experiment must "
        "keep the token routing distribution from the real model, then measure "
        "how the network scheduler handles dispatch and combine traffic under "
        "expert capacity constraints and remote expert placement. "
    ),
    (
        "The Switch Transformer uses top-one routing. The router computes a "
        "score for each expert and sends the token to the selected expert if "
        "capacity remains. Dropped tokens, hot experts, and uneven source to "
        "destination traffic are exactly the effects that make MoE all-to-all "
        "different from a uniform synthetic exchange. "
    ),
    (
        "Qinle Torus DimRotation divides each payload into dimension chunks. "
        "Different chunks start from different dimensions and rotate through "
        "the remaining dimensions. This keeps several torus dimensions busy "
        "at the same time, so the experiment should expose whether real MoE "
        "routing still leaves congestion on a subset of network links. "
    ),
]


def module_by_path(root: Any, dotted_path: str) -> Any:
    module: Any = root
    for part in dotted_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def expand_texts(texts: list[str], nodes: int) -> list[str]:
    if not texts:
        raise ValueError("at least one input text is required")
    return [texts[idx % len(texts)] for idx in range(nodes)]


def lengthen_text(tokenizer: Any, text: str, min_tokens: int) -> str:
    expanded = text
    while len(tokenizer(expanded, add_special_tokens=True)["input_ids"]) < min_tokens:
        expanded += " " + text
    return expanded


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_TEXTS
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump real Switch HF MoE router hidden states.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--ary", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=128)
    parser.add_argument(
        "--router-module",
        default="encoder.block.1.layer.1.mlp.router.classifier",
        help="module whose forward input is captured",
    )
    parser.add_argument(
        "--router-key",
        default="encoder.block.1.layer.1.mlp.router.classifier.weight",
        help="state_dict key consumed by the server-side PyTorchSim driver",
    )
    parser.add_argument("--texts-file", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

    model_dir = Path(args.model_dir)
    nodes = args.ary ** args.dims
    out_path = Path(args.out) if args.out else model_dir / (
        f"switch_base8_encoder_block1_hidden_{args.ary}x"
        f"{'x'.join([str(args.ary)] * (args.dims - 1))}_tok{args.tokens_per_source}.pt"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.eval()

    base_texts = load_texts(Path(args.texts_file) if args.texts_file else None)
    texts = [
        lengthen_text(tokenizer, text, args.tokens_per_source)
        for text in expand_texts(base_texts, nodes)
    ]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.tokens_per_source,
    )

    captured: list[torch.Tensor] = []
    router_module = module_by_path(model, args.router_module)

    def capture_input(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().cpu())

    handle = router_module.register_forward_pre_hook(capture_input)
    with torch.no_grad():
        model.encoder(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            output_router_logits=True,
            return_dict=True,
        )
    handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one captured router input, got {len(captured)}")

    hidden = captured[0]
    selected = []
    source_token_counts = []
    for src in range(nodes):
        valid = encoded.attention_mask[src].bool()
        tokens = hidden[src][valid]
        if tokens.shape[0] < args.tokens_per_source:
            raise RuntimeError(
                f"source {src} only has {tokens.shape[0]} non-pad tokens; "
                f"need {args.tokens_per_source}"
            )
        selected.append(tokens[: args.tokens_per_source])
        source_token_counts.append(int(tokens.shape[0]))

    hidden_by_source = torch.stack(selected, dim=0).contiguous()
    router_weight = module_by_path(model, args.router_module).weight.detach().cpu()
    logits = hidden_by_source.reshape(-1, hidden_by_source.shape[-1]) @ router_weight.t()
    top1 = torch.argmax(logits, dim=-1)

    payload = {
        "model_id": "google/switch-base-8",
        "model_dir": str(model_dir),
        "router_module": args.router_module,
        "router_key": args.router_key,
        "nodes": nodes,
        "dims": args.dims,
        "ary": args.ary,
        "tokens_per_source": args.tokens_per_source,
        "d_model": int(model.config.d_model),
        "num_experts": int(model.config.num_experts),
        "expert_capacity": int(model.config.expert_capacity),
        "hidden_states": hidden_by_source.float(),
        "input_ids": encoded.input_ids.cpu(),
        "attention_mask": encoded.attention_mask.cpu(),
        "source_token_counts": source_token_counts,
        "texts": texts,
        "top1_counts_before_capacity": torch.bincount(
            top1,
            minlength=int(model.config.num_experts),
        ).tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"wrote hidden states -> {out_path}")
    print(f"hidden_states shape : {tuple(hidden_by_source.shape)}")
    print(f"router weight shape : {tuple(router_weight.shape)}")
    print(f"top1 counts         : {payload['top1_counts_before_capacity']}")


if __name__ == "__main__":
    main()
