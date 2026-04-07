"""
Train ablation-model language models on WikiText with a clean CLI entrypoint.

This script deliberately uses `layer_depth_attention.ablation_models.TinyDecoderLM`
instead of the main `model.py` stack so the user can run controlled attention
ablations on a single, explicit code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from layer_depth_attention.ablation_models import TinyDecoderLM

try:
    import swanlab
except Exception:
    swanlab = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ablation models on WikiText.")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--data-source", choices=["hf", "local_text"], default="hf")
    parser.add_argument("--local-data-dir", default="")
    parser.add_argument("--add-eos-between-lines", choices=["on", "off"], default="on")

    parser.add_argument("--max-seq-len", type=int, default=256) #默认256
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie-weights", choices=["on", "off"], default="on")
    parser.add_argument("--use-pos-emb", choices=["on", "off"], default="on")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--final-test-batches", type=int, default=0)
    parser.add_argument("--train-loss-window", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-data-parallel", choices=["on", "off"], default="on")
    parser.add_argument("--val-use-cursor", choices=["on", "off"], default="off")

    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
                # "baseline",
                #   "attn_residual",
                    # "attn_residual_2d"
                    # "shared_kv_depth_memory_dualq_sublayer"
                    "light_attention"


                    ],
        help="Attention variants defined in ablation_models.TinyDecoderLM",
    )

    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "wikitext103_ablation"))
    parser.add_argument("--log-backend", choices=["none", "swanlab"], default="swanlab")
    parser.add_argument("--log-project", default="Layer-Depth-Attention-WikiText103-Ablation")
    parser.add_argument("--log-workspace", default="")
    parser.add_argument("--run-note", default="")
    return parser.parse_args()


def as_bool(flag: str) -> bool:
    return flag == "on"


@dataclass
class DataConfig:
    data_source: str
    local_data_dir: str
    dataset_name: str
    dataset_config: str
    text_field: str
    tokenizer_name: str
    max_seq_len: int
    add_eos_between_lines: bool


class LMData:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 100_000_000
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.eos_token_id
        self.local_data_dir = Path(cfg.local_data_dir) if cfg.local_data_dir else None
        self.dataset = None if cfg.data_source == "local_text" else load_dataset(cfg.dataset_name, cfg.dataset_config)

        self.splits = self._load_or_tokenize()
        self.eval_windows = {
            split: self._build_eval_windows(self.splits[split]) for split in ("validation", "test")
        }
        self.eval_cursors = {"validation": 0, "test": 0}

    def _cache_path(self) -> Path:
        cache_dir = PROJECT_ROOT / ".cache" / "lmdata"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = {
            "source": self.cfg.data_source,
            "local_data_dir": self.cfg.local_data_dir,
            "dataset_name": self.cfg.dataset_name,
            "dataset_config": self.cfg.dataset_config,
            "text_field": self.cfg.text_field,
            "tokenizer_name": self.cfg.tokenizer_name,
            "add_eos_between_lines": self.cfg.add_eos_between_lines,
        }
        digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"{digest}.pt"

    def _load_or_tokenize(self) -> Dict[str, torch.Tensor]:
        cache_path = self._cache_path()
        if cache_path.exists():
            print(f"[data] loading cache: {cache_path}")
            return torch.load(cache_path, weights_only=True)
        print("[data] tokenizing raw text splits...")
        splits = self._tokenize_all_splits()
        torch.save(splits, cache_path)
        print(f"[data] saved cache: {cache_path}")
        return splits

    def _read_split_text(self, split_name: str) -> str:
        if self.cfg.data_source == "local_text":
            if self.local_data_dir is None:
                raise ValueError("local_data_dir is required when data_source=local_text")
            path = self.local_data_dir / f"{split_name}.txt"
            return path.read_text("utf-8")
        texts = self.dataset[split_name][self.cfg.text_field]
        return "\n".join(texts)

    def _tokenize_split(self, split_name: str) -> torch.Tensor:
        text = self._read_split_text(split_name)
        if self.cfg.add_eos_between_lines:
            eos = self.tokenizer.eos_token
            text = text.replace("\n", f" {eos} ")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return torch.tensor(token_ids, dtype=torch.long)

    def _tokenize_all_splits(self) -> Dict[str, torch.Tensor]:
        return {split: self._tokenize_split(split) for split in ("train", "validation", "test")}

    def _build_eval_windows(self, token_ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        chunk = self.cfg.max_seq_len + 1
        starts = range(0, token_ids.numel() - chunk + 1, self.cfg.max_seq_len)
        return [(token_ids[s:s + chunk][:-1], token_ids[s:s + chunk][1:]) for s in starts]

    def sample_train_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.splits["train"]
        chunk = self.cfg.max_seq_len + 1
        starts = torch.randint(0, token_ids.numel() - chunk + 1, (batch_size,))
        batch = torch.stack([token_ids[s:s + chunk] for s in starts.tolist()]).to(device)
        return batch[:, :-1], batch[:, 1:]

    def iter_eval_batches(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        max_batches: Optional[int],
        use_cursor: bool,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        windows = self.eval_windows[split]
        total = len(windows)
        if total == 0:
            return
        batches_to_yield = math.ceil(total / batch_size) if max_batches is None else max_batches
        cursor = self.eval_cursors[split] if use_cursor else 0
        for _ in range(batches_to_yield):
            batch = []
            for _ in range(batch_size):
                batch.append(windows[cursor])
                cursor = (cursor + 1) % total
                if max_batches is None and cursor == 0:
                    break
            xs = torch.stack([item[0] for item in batch]).to(device)
            ys = torch.stack([item[1] for item in batch]).to(device)
            yield xs, ys
            if max_batches is None and cursor == 0:
                break
        if use_cursor:
            self.eval_cursors[split] = cursor


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "norm" not in name.lower():
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_scale: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)


class SwanLabMonitor:
    def __init__(self, backend: str, project: str, experiment_name: str, config: dict, workspace: str):
        self.enabled = backend == "swanlab" and swanlab is not None
        self.project = project
        self.experiment_name = experiment_name
        self.config = config
        self.workspace = workspace

    def init(self) -> None:
        if not self.enabled:
            print("[swanlab] disabled")
            return
        try:
            api_key = os.environ.get("SWANLAB_API_KEY")
            if api_key:
                swanlab.login(api_key=api_key)
            kwargs = {
                "project": self.project,
                "experiment_name": self.experiment_name,
                "config": self.config,
            }
            if self.workspace:
                kwargs["workspace"] = self.workspace
            swanlab.init(**kwargs)
            print(f"[swanlab] init succeeded: project={self.project} experiment={self.experiment_name}")
        except Exception as exc:
            self.enabled = False
            print(f"[swanlab] init failed: {exc!r}")

    def log(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        try:
            swanlab.log(metrics, step=step)
        except Exception as exc:
            print(f"[swanlab] log failed: {exc!r}")

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            swanlab.finish()
        except Exception as exc:
            print(f"[swanlab] finish failed: {exc!r}")


def build_experiment_name(method_name: str, args: argparse.Namespace) -> str:
    return f"{method_name}_{args.dataset_config}_{args.d_model}d_{args.num_layers}l"


def build_monitor_config(
    args: argparse.Namespace,
    method_name: str,
    model_params: Optional[int] = None,
    startup_only: bool = False,
) -> Dict[str, object]:
    config: Dict[str, object] = {
        "method_name": method_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "max_seq_len": args.max_seq_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "final_test_batches": None if args.final_test_batches == 0 else args.final_test_batches,
        "data_source": args.data_source,
        "run_note": args.run_note,
        "startup_only": startup_only,
    }
    if model_params is not None:
        config["model_params"] = model_params
    return config


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: LMData,
    split: str,
    batch_size: int,
    eval_batches: Optional[int],
    device: torch.device,
    use_cursor: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inputs, labels in data.iter_eval_batches(split, batch_size, device, eval_batches, use_cursor):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}


def run_experiment(
    method_name: str,
    args: argparse.Namespace,
    data: LMData,
    device: torch.device,
    monitor: Optional[SwanLabMonitor] = None,
) -> Dict[str, object]:
    model = TinyDecoderLM(
        vocab_size=data.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_type=method_name,
        tie_weights=as_bool(args.tie_weights),
        use_pos_emb=as_bool(args.use_pos_emb),
    )
    if device.type == "cuda" and as_bool(args.use_data_parallel) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    param_count = sum(p.numel() for p in raw_model.parameters())
    print(f"[{method_name}] model_params={param_count}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    recent_losses = deque(maxlen=args.train_loss_window)
    history: List[Dict[str, float]] = []
    best_record = None
    best_state = None
    started = time.perf_counter()

    owns_monitor = monitor is None
    if monitor is None:
        monitor = SwanLabMonitor(
            backend=args.log_backend,
            project=args.log_project,
            experiment_name=build_experiment_name(method_name, args),
            config=build_monitor_config(args, method_name, model_params=param_count),
            workspace=args.log_workspace,
        )
    else:
        monitor.config.update(build_monitor_config(args, method_name, model_params=param_count))

    if owns_monitor:
        monitor.init()
    monitor.log({"model_params": param_count}, step=0)

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup_steps, args.min_lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            inputs, labels = data.sample_train_batch(args.batch_size, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            (loss / args.grad_accum_steps).backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss = step_loss / args.grad_accum_steps
        recent_losses.append(train_loss)

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            val = evaluate(
                model,
                data,
                "validation",
                args.batch_size,
                args.eval_batches,
                device,
                use_cursor=as_bool(args.val_use_cursor),
            )
            elapsed = time.perf_counter() - started
            record = {
                "step": step,
                "lr": lr,
                "train_loss_raw": train_loss,
                "train_loss_avg": sum(recent_losses) / len(recent_losses),
                "val_loss": val["loss"],
                "val_ppl": val["perplexity"],
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60.0,
            }
            history.append(record)
            if best_record is None or record["val_loss"] < best_record["val_loss"]:
                best_record = dict(record)
                best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            monitor.log(record, step=step)
            print(
                f"method={method_name} step={step} lr={lr:.6f} "
                f"train_loss_avg={record['train_loss_avg']:.4f} "
                f"val_loss={record['val_loss']:.4f} val_ppl={record['val_ppl']:.2f} "
                f"elapsed_min={record['elapsed_minutes']:.2f}"
            )

    if best_state is not None:
        raw_model.load_state_dict(best_state)
    final_test_batches = None if args.final_test_batches == 0 else args.final_test_batches
    test = evaluate(model, data, "test", args.batch_size, final_test_batches, device, use_cursor=False)

    summary = {
        "method": method_name,
        "model_params": param_count,
        "best_step": best_record["step"] if best_record else None,
        "best_val_loss": best_record["val_loss"] if best_record else None,
        "best_val_ppl": best_record["val_ppl"] if best_record else None,
        "best_test_loss": test["loss"],
        "best_test_ppl": test["perplexity"],
        "history": history,
        "args": vars(args),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_name}_{args.dataset_config}_{args.d_model}d_{args.num_layers}l.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    monitor.log({"best_test_loss": summary["best_test_loss"], "best_test_ppl": summary["best_test_ppl"]}, step=args.steps)
    if owns_monitor:
        monitor.finish()

    print(f"best_step={summary['best_step']} best_val_loss={summary['best_val_loss']:.4f} best_val_ppl={summary['best_val_ppl']:.2f}")
    print(f"best_test_loss={summary['best_test_loss']:.4f} best_test_ppl={summary['best_test_ppl']:.2f}")
    print(f"saved metrics to {out_path}")
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} num_gpus = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    startup_monitor: Optional[SwanLabMonitor] = None
    single_method = len(args.methods) == 1
    if single_method:
        method_name = args.methods[0]
        startup_monitor = SwanLabMonitor(
            backend=args.log_backend,
            project=args.log_project,
            experiment_name=build_experiment_name(method_name, args),
            config=build_monitor_config(args, method_name, startup_only=True),
            workspace=args.log_workspace,
        )
        startup_monitor.init()
        startup_monitor.log({"program_started": 1, "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0}, step=0)

    data = LMData(
        DataConfig(
            data_source=args.data_source,
            local_data_dir=args.local_data_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_field=args.text_field,
            tokenizer_name=args.tokenizer_name,
            max_seq_len=args.max_seq_len,
            add_eos_between_lines=as_bool(args.add_eos_between_lines),
        )
    )
    print(f"vocab_size = {data.vocab_size}")
    for split_name, token_ids in data.splits.items():
        print(f"{split_name} tokens = {token_ids.numel()}")
    if startup_monitor is not None:
        startup_monitor.log(
            {
                "vocab_size": data.vocab_size,
                "train_tokens": data.splits["train"].numel(),
                "validation_tokens": data.splits["validation"].numel(),
                "test_tokens": data.splits["test"].numel(),
            },
            step=0,
        )

    results = {}
    for method_name in args.methods:
        print("\n" + "=" * 80)
        print(f"running {method_name}")
        print("=" * 80)
        method_monitor = startup_monitor if (startup_monitor is not None and method_name == args.methods[0]) else None
        results[method_name] = run_experiment(method_name, args, data, device, monitor=method_monitor)

    print("\n" + "=" * 80)
    print("summary")
    print("=" * 80)
    for name, result in sorted(results.items(), key=lambda item: item[1]["best_test_ppl"]):
        print(
            f"{name:<40} val_ppl={result['best_val_ppl']:.2f} "
            f"test_ppl={result['best_test_ppl']:.2f} params={result['model_params'] / 1e6:.2f}M"
        )


if __name__ == "__main__":
    main()
