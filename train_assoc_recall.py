import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from layer_depth_attention.data import AssocRecallConfig, AssociativeRecallDataset
from layer_depth_attention.model import TinyDecoderLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument(
        "--attention-type",
        choices=["baseline", "depth_memory", "depth_memory_value_reproj", "attn_residuals", "attn_residuals_value_reproj"],
        default="baseline",
    )
    parser.add_argument("--attn-residual", choices=["on", "off"], default="on")
    parser.add_argument("--ffn-residual", choices=["on", "off"], default="on")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: TinyDecoderLM,
    dataset: AssociativeRecallDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for _ in range(eval_batches):
        inputs, labels = dataset.sample_batch(batch_size)
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
        preds = logits[:, -1, :].argmax(dim=-1)
        targets = labels[:, -1]
        correct += (preds == targets).sum().item()
        total += targets.numel()
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    cfg = AssocRecallConfig(vocab_size=args.vocab_size, num_pairs=args.num_pairs)
    train_data = AssociativeRecallDataset(cfg, seed=args.seed)
    eval_data = AssociativeRecallDataset(cfg, seed=args.seed + 1)

    model = TinyDecoderLM(
        vocab_size=args.vocab_size,
        max_seq_len=cfg.sequence_length - 1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attention_type=args.attention_type,
        attn_residual=args.attn_residual == "on",
        ffn_residual=args.ffn_residual == "on",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []

    for step in range(1, args.steps + 1):
        inputs, labels = train_data.sample_batch(args.batch_size)
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            metrics = evaluate(model, eval_data, args.batch_size, args.eval_batches, device)
            metrics["step"] = step
            metrics["train_loss"] = loss.item()
            history.append(metrics)
            print(
                f"step={step} train_loss={loss.item():.4f} "
                f"eval_loss={metrics['loss']:.4f} eval_acc={metrics['accuracy']:.4f}"
            )

    attn_tag = "attnres" if args.attn_residual == "on" else "noattnres"
    ffn_tag = "ffnres" if args.ffn_residual == "on" else "noffnres"
    output_name = args.output or f"artifacts/assoc_recall_{args.attention_type}_{attn_tag}_{ffn_tag}.json"
    output_path = ROOT / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved metrics to {output_path}")


if __name__ == "__main__":
    main()
