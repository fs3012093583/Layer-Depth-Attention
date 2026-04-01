import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from layer_depth_attention.lm_data import WikiTextLMConfig, WikiTextLanguageModelingData
from layer_depth_attention.model import TinyDecoderLM
from layer_depth_attention.experiment_logging import build_monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default="external_data/wikitext-2-raw-v1")
    parser.add_argument("--tokenizer-dir", default="external_data/gpt2_tokenizer")
    parser.add_argument("--model-preset", choices=["none", "wt2_standard_a"], default="none")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--attention-type",
        choices=[
            "baseline",
            "dual_axis_memory",
            "dual_axis_full",
            "depth_memory",
            "depth_memory_2d_prefix",
            "depth_memory_qkv_reproj",
            "depth_memory_value_reproj",
            "depth_memory_value_reproj_normed",
            "depth_memory_directkv_dualq",
            "depth_memory_directkv_qmix",
            "depth_memory_value_reproj_normed_ffn_qattn",
            "depth_memory_value_reproj_normed_dualq_ffn_qattn_dualq",
            "depth_memory_value_reproj_dualq",
            "attn_residuals",
            "attn_residuals_dual_axis",
            "attn_residuals_value_reproj",
            "attn_residuals_value_reproj_normed",
            "attn_residuals_moe",
        ],
        default="baseline",
    )
    parser.add_argument("--attn-residual", choices=["on", "off"], default="on")
    parser.add_argument("--ffn-residual", choices=["on", "off"], default="on")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-backend", choices=["none", "swanlab"], default="none")
    parser.add_argument("--log-project", default="Layer-Depth-Attention")
    parser.add_argument("--log-experiment-name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_model_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_preset == "wt2_standard_a":
        args.seq_len = 512
        args.d_model = 384
        args.num_layers = 6
        args.num_heads = 8
        args.mlp_ratio = 4
        args.dropout = 0.2
        args.batch_size = 4
        args.grad_accum_steps = 2
        args.steps = max(args.steps, 2000)
        args.eval_interval = 400
        args.eval_batches = 20
        args.lr = 3e-4
        args.min_lr_scale = 0.1
        args.warmup_steps = 100
        args.weight_decay = 0.01
    return args


def build_optimizer(model: TinyDecoderLM, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "norm" not in name:
            decay_params.append(param)
        else:
            nodecay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_scale: float) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)


@torch.no_grad()
def evaluate(
    model: TinyDecoderLM,
    data: WikiTextLanguageModelingData,
    split: str,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for inputs, labels in data.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        max_batches=eval_batches,
    ):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        token_count = labels.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count
        batches += 1
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "batches": batches,
    }


def main() -> None:
    args = parse_args()
    args = apply_model_preset(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = WikiTextLanguageModelingData(
        WikiTextLMConfig(
            data_dir=ROOT / args.data_dir,
            tokenizer_dir=ROOT / args.tokenizer_dir,
            seq_len=args.seq_len,
        )
    )

    model = TinyDecoderLM(
        vocab_size=data.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_type=args.attention_type,
        num_experts=args.num_experts,
        attn_residual=args.attn_residual == "on",
        ffn_residual=args.ffn_residual == "on",
        tie_weights=True,
    ).to(device)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"model_params={param_count}")

    output_name = args.output or f"artifacts/wikitext2_{args.attention_type}_{'attnres' if args.attn_residual == 'on' else 'noattnres'}_{'ffnres' if args.ffn_residual == 'on' else 'noffnres'}.json"
    output_path = ROOT / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = checkpoint_dir / f"{output_path.stem}_latest.pt"
    best_ckpt_path = checkpoint_dir / f"{output_path.stem}_best.pt"
    default_experiment_name = Path(output_name).stem
    monitor = build_monitor(
        backend=args.log_backend,
        project=args.log_project,
        experiment_name=args.log_experiment_name or default_experiment_name,
    )
    run_config = vars(args).copy()
    run_config["resolved_output"] = str(ROOT / output_name)
    run_config["model_params"] = param_count
    monitor.init_experiment(run_config)
    monitor.log_metrics({"model_params": param_count}, step=0)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    history = []
    best_record = None
    best_model_state = None
    train_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup_steps, args.min_lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(args.grad_accum_steps):
            inputs, labels = data.sample_train_batch(args.batch_size, device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            val_metrics = evaluate(model, data, "validation", args.batch_size, args.eval_batches, device)
            elapsed_seconds = time.perf_counter() - train_start
            record = {
                "step": step,
                "lr": lr,
                "train_loss": running_loss / args.grad_accum_steps,
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["perplexity"],
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
            }
            history.append(record)
            latest_payload = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "record": record,
                "config": vars(args),
                "model_params": param_count,
            }
            torch.save(latest_payload, latest_ckpt_path)
            if best_record is None or record["val_loss"] < best_record["val_loss"]:
                best_record = copy.deepcopy(record)
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(
                    {
                        "step": step,
                        "model_state": best_model_state,
                        "record": best_record,
                        "config": vars(args),
                        "model_params": param_count,
                    },
                    best_ckpt_path,
                )
            monitor.log_metrics(record, step=step)
            print(
                f"step={step} lr={lr:.6f} train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f} val_ppl={record['val_ppl']:.2f} "
                f"elapsed_min={record['elapsed_minutes']:.2f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    best_test_metrics = evaluate(model, data, "test", args.batch_size, args.eval_batches, device)
    summary = {
        "history": history,
        "best_step": None if best_record is None else best_record["step"],
        "best_val_loss": None if best_record is None else best_record["val_loss"],
        "best_val_ppl": None if best_record is None else best_record["val_ppl"],
        "best_test_loss": best_test_metrics["loss"],
        "best_test_ppl": best_test_metrics["perplexity"],
        "best_checkpoint_path": str(best_ckpt_path),
        "latest_checkpoint_path": str(latest_ckpt_path),
        "config": vars(args),
    }
    monitor.log_metrics(
        {
            "best_test_loss": summary["best_test_loss"],
            "best_test_ppl": summary["best_test_ppl"],
        },
        step=args.steps,
    )
    print(
        f"best_step={summary['best_step']} best_val_loss={summary['best_val_loss']:.4f} "
        f"best_val_ppl={summary['best_val_ppl']:.2f}"
    )
    print(
        f"best_test_loss={summary['best_test_loss']:.4f} "
        f"best_test_ppl={summary['best_test_ppl']:.2f}"
    )

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved metrics to {output_path}")
    monitor.finish()


if __name__ == "__main__":
    main()
