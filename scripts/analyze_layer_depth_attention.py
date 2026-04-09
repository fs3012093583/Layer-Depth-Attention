"""
Analyze row-vs-depth attention usage from a saved ablation-model checkpoint.

This script loads a checkpoint produced by `train_wikitext103_ablation.py`,
rebuilds the model and data pipeline, instruments the layer-depth attention
modules, and records:

- average row attention mass per layer
- average depth attention mass per layer
- average depth-memory allocation per memory slot

It also saves simple plots for paper use.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from layer_depth_attention.ablation_models import (  # noqa: E402
    BaselineAttention,
    DepthMemoryReuseRowQKVAttention,
    SharedKVDepthMemoryDualQAttention,
    TinyDecoderLM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved layer-depth attention checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0, help="Override checkpoint batch size for analysis.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cache-file", default="", help="Optional cached lmdata .pt file to bypass tokenizer/dataset loading.")
    parser.add_argument("--analysis-seq-len", type=int, default=0, help="Optional shorter sequence length for analysis only.")
    parser.add_argument("--matrix-layer", type=int, default=-1, help="1-based layer index to export attention matrices for; <=0 disables.")
    parser.add_argument("--matrix-sample-index", type=int, default=0, help="Sample index within the analyzed batch.")
    return parser.parse_args()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def infer_cache_path(train_args: argparse.Namespace) -> Path:
    cache_dir = PROJECT_ROOT / ".cache" / "lmdata"
    key = {
        "source": train_args.data_source,
        "local_data_dir": train_args.local_data_dir,
        "dataset_name": train_args.dataset_name,
        "dataset_config": train_args.dataset_config,
        "text_field": train_args.text_field,
        "tokenizer_name": train_args.tokenizer_name,
        "add_eos_between_lines": as_bool(train_args.add_eos_between_lines),
    }
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}.pt"


def iter_cached_eval_batches(
    split_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    total_windows = (split_tokens.numel() - 1) // seq_len
    if total_windows <= 0:
        return
    cursor = 0
    produced = 0
    while produced < max_batches and cursor < total_windows:
        cur_batch = min(batch_size, total_windows - cursor)
        xs = []
        ys = []
        for i in range(cur_batch):
            start = (cursor + i) * seq_len
            x = split_tokens[start : start + seq_len]
            y = split_tokens[start + 1 : start + seq_len + 1]
            xs.append(x)
            ys.append(y)
        cursor += cur_batch
        produced += 1
        yield torch.stack(xs).to(device), torch.stack(ys).to(device)


class AttentionRecorder:
    def __init__(self, num_layers: int, max_slots: int):
        self.num_layers = num_layers
        self.max_slots = max_slots
        self.row_mass_sum = np.zeros(num_layers, dtype=np.float64)
        self.depth_mass_sum = np.zeros(num_layers, dtype=np.float64)
        self.count_sum = np.zeros(num_layers, dtype=np.float64)
        self.depth_slot_sum = np.zeros((num_layers, max_slots), dtype=np.float64)
        self.depth_slot_count = np.zeros((num_layers, max_slots), dtype=np.float64)
        self.depth_pos_sum: Optional[np.ndarray] = None
        self.depth_pos_count: Optional[np.ndarray] = None

    def record(self, layer_idx: int, row_w: torch.Tensor, dep_w: Optional[torch.Tensor]) -> None:
        row_mass = row_w.sum(dim=-1)  # [B, H, S]
        self.row_mass_sum[layer_idx] += row_mass.sum().item()
        self.count_sum[layer_idx] += row_mass.numel()

        if dep_w is None:
            return

        depth_mass = dep_w.sum(dim=-1)  # [B, H, S]
        self.depth_mass_sum[layer_idx] += depth_mass.sum().item()
        self.count_sum[layer_idx] += 0.0

        depth_pos = depth_mass.sum(dim=(0, 1)).detach().cpu().numpy()  # [S]
        seq_len = depth_pos.shape[0]
        if self.depth_pos_sum is None:
            self.depth_pos_sum = np.zeros((self.num_layers, seq_len), dtype=np.float64)
            self.depth_pos_count = np.zeros((self.num_layers, seq_len), dtype=np.float64)
        self.depth_pos_sum[layer_idx, :seq_len] += depth_pos
        self.depth_pos_count[layer_idx, :seq_len] += depth_mass.shape[0] * depth_mass.shape[1]

        slot_mass = dep_w.sum(dim=(0, 1, 2)).detach().cpu().numpy()  # [Lmem]
        slots = slot_mass.shape[0]
        self.depth_slot_sum[layer_idx, :slots] += slot_mass
        self.depth_slot_count[layer_idx, :slots] += row_mass.numel()

    def summary(self) -> Dict[str, object]:
        row_mass_mean = np.divide(
            self.row_mass_sum,
            np.maximum(self.count_sum, 1.0),
        )
        depth_mass_mean = np.divide(
            self.depth_mass_sum,
            np.maximum(self.count_sum, 1.0),
        )
        depth_slot_mean = np.divide(
            self.depth_slot_sum,
            np.maximum(self.depth_slot_count, 1.0),
        )
        depth_slot_mean[self.depth_slot_count == 0] = np.nan
        available_slots = np.sum(self.depth_slot_count > 0, axis=1).astype(np.float64)
        depth_mass_per_slot_mean = np.divide(
            depth_mass_mean,
            np.maximum(available_slots, 1.0),
        )
        depth_mass_per_slot_mean[available_slots == 0] = np.nan
        if self.depth_pos_sum is not None and self.depth_pos_count is not None:
            depth_pos_mean = np.divide(
                self.depth_pos_sum,
                np.maximum(self.depth_pos_count, 1.0),
            )
            depth_pos_mean[self.depth_pos_count == 0] = np.nan
            depth_pos_mean_list = depth_pos_mean.tolist()
        else:
            depth_pos_mean_list = []
        return {
            "row_mass_mean": row_mass_mean.tolist(),
            "depth_mass_mean": depth_mass_mean.tolist(),
            "available_slots": available_slots.tolist(),
            "depth_mass_per_slot_mean": depth_mass_per_slot_mean.tolist(),
            "depth_slot_mean": depth_slot_mean.tolist(),
            "depth_pos_mean": depth_pos_mean_list,
        }


class MatrixCapture:
    def __init__(self, target_layer_idx: int, sample_index: int):
        self.target_layer_idx = target_layer_idx
        self.sample_index = sample_index
        self.captured = False
        self.payload: Dict[str, object] = {}

    def maybe_record(self, layer_idx: int, row_w: torch.Tensor, dep_w: Optional[torch.Tensor]) -> None:
        if self.captured or layer_idx != self.target_layer_idx:
            return
        if row_w.size(0) <= self.sample_index:
            return

        row = row_w[self.sample_index].mean(dim=0).detach().cpu().numpy()  # [S, S]
        dep = None
        joint = row
        if dep_w is not None:
            dep = dep_w[self.sample_index].mean(dim=0).detach().cpu().numpy()  # [S, Lmem]
            joint = np.concatenate([row, dep], axis=1)

        def row_normalized_entropy(mat: np.ndarray) -> float:
            probs = np.clip(mat, 1e-12, None)
            probs = probs / probs.sum(axis=1, keepdims=True)
            ent = -(probs * np.log(probs)).sum(axis=1)
            max_ent = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
            return float(np.mean(ent / max_ent))

        self.payload = {
            "layer_index_1based": layer_idx + 1,
            "sample_index": self.sample_index,
            "row_matrix": row.tolist(),
            "depth_matrix": None if dep is None else dep.tolist(),
            "joint_matrix": joint.tolist(),
            "row_normalized_entropy": row_normalized_entropy(row),
            "joint_normalized_entropy": row_normalized_entropy(joint),
        }
        self.captured = True


def instrument_model(model: TinyDecoderLM, recorder: AttentionRecorder, matrix_capture: Optional[MatrixCapture] = None) -> None:
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn

        if isinstance(attn, BaselineAttention):
            def make_forward(module: BaselineAttention, idx: int):
                def forward(
                    x: torch.Tensor,
                    past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    past_x: Optional[torch.Tensor] = None,
                ):
                    q, k, v = module.qkv_proj(x).chunk(3, dim=-1)
                    q = module.split_heads(q)
                    k = module.split_heads(k)
                    v = module.split_heads(v)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(module.head_dim)
                    scores = scores.masked_fill(module.causal_mask(x.size(1), x.device), float("-inf"))
                    row_w = module.dropout(torch.softmax(scores, dim=-1))
                    out = torch.matmul(row_w, v)

                    recorder.record(idx, row_w, None)
                    if matrix_capture is not None:
                        matrix_capture.maybe_record(idx, row_w, None)
                    return module.out_proj(module.merge_heads(out)), (k, v)

                return forward

            attn.forward = make_forward(attn, layer_idx)

        elif isinstance(attn, SharedKVDepthMemoryDualQAttention):
            def make_forward(module: SharedKVDepthMemoryDualQAttention, idx: int):
                def forward(
                    x: torch.Tensor,
                    past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    past_x: Optional[torch.Tensor] = None,
                ):
                    q_row, k_row, v_row = module.qkv_row_proj(x).chunk(3, dim=-1)
                    q_row = module.split_heads(q_row)
                    k_row = module.split_heads(k_row)
                    v_row = module.split_heads(v_row)

                    q_col = q_row
                    k_col = module.split_heads(module.shared_memory_kv.k_proj(x))
                    v_col = module.split_heads(module.shared_memory_kv.v_proj(x))

                    seq_len = x.size(1)
                    token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(module.head_dim)
                    token_scores = token_scores.masked_fill(
                        module.causal_mask(seq_len, x.device), float("-inf")
                    )

                    dep_w = None
                    if past_kv is not None:
                        past_keys, past_values = past_kv
                        memory_scores = torch.einsum(
                            "b h s d, b h s l d -> b h s l", q_col, past_keys
                        ) / math.sqrt(module.head_dim)
                        scores = torch.cat([token_scores, memory_scores], dim=-1)
                        weights = module.dropout(torch.softmax(scores, dim=-1))
                        row_w = weights[..., :seq_len]
                        dep_w = weights[..., seq_len:]
                        token_ctx = torch.matmul(row_w, v_row)
                        mem_ctx = torch.einsum("b h s l, b h s l d -> b h s d", dep_w, past_values)
                        out = token_ctx + mem_ctx
                    else:
                        row_w = module.dropout(torch.softmax(token_scores, dim=-1))
                        out = torch.matmul(row_w, v_row)

                    recorder.record(idx, row_w, dep_w)
                    if matrix_capture is not None:
                        matrix_capture.maybe_record(idx, row_w, dep_w)
                    return module.out_proj(module.merge_heads(out)), (k_col, v_col)

                return forward

            attn.forward = make_forward(attn, layer_idx)

        elif isinstance(attn, DepthMemoryReuseRowQKVAttention):
            def make_forward(module: DepthMemoryReuseRowQKVAttention, idx: int):
                def forward(
                    x: torch.Tensor,
                    past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    past_x: Optional[torch.Tensor] = None,
                ):
                    q_row, k_row, v_row = module.qkv_row_proj(x).chunk(3, dim=-1)
                    q_row = module.split_heads(q_row)
                    k_row = module.split_heads(k_row)
                    v_row = module.split_heads(v_row)

                    q_col = q_row
                    seq_len = x.size(1)
                    token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(module.head_dim)
                    token_scores = token_scores.masked_fill(
                        module.causal_mask(seq_len, x.device), float("-inf")
                    )

                    dep_w = None
                    if past_kv is not None:
                        past_keys, past_values = past_kv
                        memory_scores = torch.einsum(
                            "b h s d, b h s l d -> b h s l", q_col, past_keys
                        ) / math.sqrt(module.head_dim)
                        scores = torch.cat([token_scores, memory_scores], dim=-1)
                        weights = module.dropout(torch.softmax(scores, dim=-1))
                        row_w = weights[..., :seq_len]
                        dep_w = weights[..., seq_len:]
                        token_ctx = torch.matmul(row_w, v_row)
                        mem_ctx = torch.einsum("b h s l, b h s l d -> b h s d", dep_w, past_values)
                        out = token_ctx + mem_ctx
                    else:
                        row_w = module.dropout(torch.softmax(token_scores, dim=-1))
                        out = torch.matmul(row_w, v_row)

                    recorder.record(idx, row_w, dep_w)
                    if matrix_capture is not None:
                        matrix_capture.maybe_record(idx, row_w, dep_w)
                    return module.out_proj(module.merge_heads(out)), (k_row, v_row)

                return forward

            attn.forward = make_forward(attn, layer_idx)


def make_plots(summary: Dict[str, object], out_dir: Path) -> None:
    row = np.array(summary["row_mass_mean"], dtype=float)
    depth = np.array(summary["depth_mass_mean"], dtype=float)
    depth_per_slot = np.array(summary.get("depth_mass_per_slot_mean", []), dtype=float)
    slots = np.array(summary.get("available_slots", []), dtype=float)
    heat = np.array(summary["depth_slot_mean"], dtype=float)
    depth_pos = np.array(summary.get("depth_pos_mean", []), dtype=float)
    layers = np.arange(1, len(row) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(layers, row, marker="o", label="Row Mass")
    plt.plot(layers, depth, marker="o", label="Depth Mass")
    plt.xlabel("Layer")
    plt.ylabel("Average Attention Mass")
    plt.title("Row vs Depth Attention Mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "row_vs_depth_mass.png", dpi=200)
    plt.close()

    if depth_per_slot.size > 0:
        plt.figure(figsize=(7, 4))
        plt.plot(layers, depth, marker="o", label="Total Depth Mass")
        plt.plot(layers, depth_per_slot, marker="o", label="Depth Mass / Available Slot")
        plt.xlabel("Layer")
        plt.ylabel("Average Attention Mass")
        plt.title("Depth Attention With Slot-Count Normalization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "depth_mass_normalized_by_slots.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(layers, slots, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Available Memory Slots")
        plt.title("Available Depth-Memory Slots by Layer")
        plt.tight_layout()
        plt.savefig(out_dir / "available_depth_slots.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Average Depth Attention Mass")
    plt.xlabel("Memory Slot Index")
    plt.ylabel("Current Layer")
    plt.title("Depth-Memory Allocation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "depth_slot_heatmap.png", dpi=200)
    plt.close()

    if depth_pos.size > 0:
        plt.figure(figsize=(8, 4.5))
        plt.imshow(depth_pos, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Average Depth Attention Mass")
        plt.xlabel("Token Position")
        plt.ylabel("Current Layer")
        plt.title("Depth Attention by Layer and Token Position")
        plt.tight_layout()
        plt.savefig(out_dir / "depth_by_position_heatmap.png", dpi=200)
        plt.close()


def save_matrix_plots(matrix_payload: Dict[str, object], out_dir: Path) -> None:
    if not matrix_payload:
        return

    row = np.array(matrix_payload["row_matrix"], dtype=float)
    depth_raw = matrix_payload.get("depth_matrix")
    depth = None if depth_raw is None else np.array(depth_raw, dtype=float)
    joint = np.array(matrix_payload["joint_matrix"], dtype=float)

    def save_heatmap(
        mat: np.ndarray,
        path: Path,
        title: str,
        xlabel: str,
        split_col: Optional[int] = None,
        left_label: str = "Row Tokens",
        right_label: str = "Depth Slots",
    ) -> None:
        plt.figure(figsize=(8, 5))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Attention Weight")
        if split_col is not None:
            plt.axvline(split_col - 0.5, color="white", linestyle="--", linewidth=1.0, alpha=0.55)
            y_text = max(0.0, mat.shape[0] * 0.06)
            left_x = max(0.0, (split_col - 1) / 2.0)
            right_x = split_col + max(0.0, (mat.shape[1] - split_col - 1) / 2.0)
            plt.text(
                left_x,
                y_text,
                left_label,
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "black", "alpha": 0.25, "pad": 2},
            )
            plt.text(
                right_x,
                y_text,
                right_label,
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "black", "alpha": 0.25, "pad": 2},
            )
            total_cols = mat.shape[1]
            right_cols = total_cols - split_col
            left_ticks = np.linspace(0, max(split_col - 1, 0), num=min(4, max(split_col, 1)))
            left_ticks = np.unique(np.round(left_ticks).astype(int))
            if right_cols > 0:
                right_ticks_local = np.linspace(0, max(right_cols - 1, 0), num=min(4, right_cols))
                right_ticks_local = np.unique(np.round(right_ticks_local).astype(int))
                right_ticks = split_col + right_ticks_local
                xticks = np.concatenate([left_ticks, right_ticks])
                xticklabels = [str(int(t)) for t in left_ticks] + [str(int(t)) for t in right_ticks_local]
            else:
                xticks = left_ticks
                xticklabels = [str(int(t)) for t in left_ticks]
            plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel)
        plt.ylabel("Query Token Position")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    layer = matrix_payload["layer_index_1based"]
    sample = matrix_payload["sample_index"]
    save_heatmap(
        row,
        out_dir / "row_attention_matrix.png",
        f"Row Attention Matrix (Layer {layer}, Sample {sample})",
        "Key Token Position",
    )
    if depth is not None:
        save_heatmap(
            depth,
            out_dir / "depth_attention_matrix.png",
            f"Depth Attention Matrix (Layer {layer}, Sample {sample})",
            "Depth Memory Slot",
        )
    save_heatmap(
        joint,
        out_dir / "joint_attention_matrix.png",
        f"Joint Attention Matrix (Layer {layer}, Sample {sample})",
        "Token Position (left) / Depth Slot Index (right)",
        split_col=row.shape[1],
        left_label="Row Tokens",
        right_label="Depth Slots",
    )


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])
    method_name = ckpt["method"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "analysis_outputs" / ckpt_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_file) if args.cache_file else infer_cache_path(train_args)
    if not cache_path.exists():
        raise FileNotFoundError(f"cached token file not found: {cache_path}")
    splits = torch.load(cache_path, map_location="cpu", weights_only=True)
    split_tokens = splits[args.split].long()

    model = TinyDecoderLM(
        vocab_size=ckpt["model_state_dict"]["token_emb.weight"].shape[0],
        max_seq_len=train_args.max_seq_len,
        d_model=train_args.d_model,
        num_layers=train_args.num_layers,
        num_heads=train_args.num_heads,
        mlp_ratio=train_args.mlp_ratio,
        dropout=train_args.dropout,
        attention_type=method_name,
        tie_weights=as_bool(train_args.tie_weights),
        use_pos_emb=as_bool(train_args.use_pos_emb),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    max_slots = train_args.num_layers * 2
    recorder = AttentionRecorder(num_layers=train_args.num_layers, max_slots=max_slots)
    matrix_capture = None
    if args.matrix_layer > 0:
        matrix_capture = MatrixCapture(target_layer_idx=args.matrix_layer - 1, sample_index=args.matrix_sample_index)
    instrument_model(model, recorder, matrix_capture=matrix_capture)

    batch_size = args.batch_size if args.batch_size > 0 else train_args.batch_size
    analysis_seq_len = args.analysis_seq_len if args.analysis_seq_len > 0 else train_args.max_seq_len

    with torch.no_grad():
        for inputs, _labels in iter_cached_eval_batches(
            split_tokens=split_tokens,
            batch_size=batch_size,
            seq_len=analysis_seq_len,
            device=device,
            max_batches=args.max_batches,
        ):
            _ = model(inputs)

    summary = recorder.summary()
    payload = {
        "checkpoint": str(ckpt_path),
        "method": method_name,
        "split": args.split,
        "max_batches": args.max_batches,
        "batch_size": batch_size,
        "analysis_seq_len": analysis_seq_len,
        **summary,
    }
    (out_dir / "attention_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(out_dir / "depth_slot_mean.npy", np.array(summary["depth_slot_mean"], dtype=float))
    make_plots(summary, out_dir)
    if matrix_capture is not None and matrix_capture.payload:
        (out_dir / "attention_matrix_summary.json").write_text(
            json.dumps(matrix_capture.payload, indent=2),
            encoding="utf-8",
        )
        save_matrix_plots(matrix_capture.payload, out_dir)

    print(f"saved analysis to {out_dir}")
    print(f"row_vs_depth plot: {out_dir / 'row_vs_depth_mass.png'}")
    print(f"depth heatmap   : {out_dir / 'depth_slot_heatmap.png'}")
    print(f"summary json    : {out_dir / 'attention_summary.json'}")


if __name__ == "__main__":
    main()
