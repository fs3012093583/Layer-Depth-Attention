"""
训练脚本 —— 消融实验
从 notebooks/103 without model.ipynb 提取，可在后台运行（不依赖 Jupyter）。

Windows 后台运行示例：
    Start-Process python -ArgumentList "scripts/train.py" `
        -RedirectStandardOutput train.log `
        -RedirectStandardError train_err.log `
        -NoNewWindow

Linux / tmux 后台运行示例：
    nohup python scripts/train.py > train.log 2>&1 &
"""

# ============================================================
# ⚙️  可改参数区（每次训练前在这里调整）
# ============================================================
CFG = {
    # 数据
    "dataset_name":          "wikitext",
    "dataset_config":        "wikitext-103-raw-v1",
    "text_field":            "text",
    "tokenizer_name":        "gpt2",
    "add_eos_between_lines": True,
    "data_source":           "hf",        # "hf" 从 HuggingFace 下载；"local_text" 用本地 txt
    "local_data_dir":        "",          # data_source="local_text" 时填本地目录

    # 模型
    "max_seq_len": 256,
    "d_model":     384,
    "num_layers":  8,
    "num_heads":   8,
    "mlp_ratio":   4,
    "dropout":     0.1,
    "tie_weights": True,
    "use_pos_emb": True,

    # 训练
    "batch_size":         8,
    "grad_accum_steps":   1,
    "steps":              40000,
    "eval_interval":      1000,
    "eval_batches":       100,    # 训练中每次验证取前 N 批
    "final_test_batches": None,   # None = 全量 test 集
    "train_loss_window":  20,
    "lr":                 3e-4,
    "min_lr_scale":       0.1,
    "warmup_steps":       100,
    "weight_decay":       0.01,
    "grad_clip":          1.0,

    # 实验方法（可以填多个，顺序执行）
    "methods_to_run": [
        "baseline",
        "attn_residual",
        "attn_residual_2d",
        # "shared_kv_baseline",
        # "shared_kv_depth_memory_dualq",
        # "shared_kv_depth_memory_dualq_sublayer",
    ],

    # 输出
    "use_data_parallel": True,
    "output_dir":        "outputs",   # 结果 JSON 保存路径（相对项目根目录）

    # SwanLab 实验追踪（不需要就把 log_backend 改成 "none"）
    "log_backend": "swanlab",
    "log_project": "Layer-Depth-Attention",
}

# ============================================================
# 依赖导入
# ============================================================
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

# 把项目 src 目录加到路径，保证 import 正常
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from layer_depth_attention.ablation_models import TinyDecoderLM

try:
    import swanlab
except Exception:
    swanlab = None

# ============================================================
# 环境初始化
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"device = {DEVICE}  num_gpus = {NUM_GPUS}")


# ============================================================
# 数据加载（支持 HuggingFace 数据集 + 本地缓存）
# ============================================================
@dataclass
class LMCfg:
    data_source:           str
    local_data_dir:        str
    dataset_name:          str
    dataset_config:        str
    text_field:            str
    tokenizer_name:        str
    max_seq_len:           int
    add_eos_between_lines: bool = True


class LMData:
    def __init__(self, cfg: LMCfg):
        self.cfg = cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 100_000_000   # 消除超长文本警告
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.eos_token_id
        self.dataset = None
        self.local_data_dir = Path(cfg.local_data_dir) if cfg.local_data_dir else Path(".")
        if cfg.data_source == "hf":
            self.dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

        self.splits = self._load_or_tokenize()
        self.eval_windows = {
            split: self._build_eval_windows(self.splits[split])
            for split in ["validation", "test"]
        }
        self.eval_cursors = {"validation": 0, "test": 0}

    # ---------- 缓存逻辑 ----------
    def _cache_path(self) -> Path:
        key = (f"{self.cfg.dataset_name}_{self.cfg.dataset_config}"
               f"_{self.cfg.tokenizer_name}_{self.cfg.add_eos_between_lines}")
        hash_str = hashlib.md5(key.encode()).hexdigest()[:10]
        cache_dir = Path(".cache/lmdata")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{hash_str}.pt"

    def _load_or_tokenize(self) -> Dict[str, torch.Tensor]:
        cache_file = self._cache_path()
        if cache_file.exists():
            print(f"⚡ 发现缓存，直接加载: {cache_file}")
            splits = torch.load(cache_file, weights_only=True)
            print("✅ 缓存加载完成！")
            return splits
        print("🔄 未发现缓存，开始分词（首次需要几分钟，之后秒开）...")
        splits = self._tokenize_all_splits()
        torch.save(splits, cache_file)
        print(f"💾 已保存缓存到: {cache_file}")
        return splits

    # ---------- 分词逻辑 ----------
    def _tokenize_split(self, split_name: str) -> torch.Tensor:
        if self.cfg.data_source == "local_text":
            texts = (self.local_data_dir / f"{split_name}.txt").read_text("utf-8").split("\n")
        else:
            texts = self.dataset[split_name][self.cfg.text_field]
        all_ids = []
        eos_id = self.tokenizer.eos_token_id
        for text in texts:
            if text.strip():
                all_ids.extend(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))
            if self.cfg.add_eos_between_lines:
                all_ids.append(eos_id)
        return torch.tensor(all_ids, dtype=torch.long)

    def _tokenize_all_splits(self) -> Dict[str, torch.Tensor]:
        return {s: self._tokenize_split(s) for s in ("train", "validation", "test")}

    # ---------- 批次采样 ----------
    def _build_eval_windows(self, token_ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        chunk = self.cfg.max_seq_len + 1
        starts = range(0, token_ids.numel() - chunk + 1, self.cfg.max_seq_len)
        return [(token_ids[s:s+chunk][:-1], token_ids[s:s+chunk][1:]) for s in starts]

    def sample_train_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.splits["train"]
        chunk = self.cfg.max_seq_len + 1
        starts = torch.randint(0, ids.numel() - chunk + 1, (batch_size,))
        batch = torch.stack([ids[s:s+chunk] for s in starts.tolist()]).to(device)
        return batch[:, :-1], batch[:, 1:]

    def iter_eval_batches(
        self, split: str, batch_size: int, device: torch.device,
        max_batches: Optional[int], use_cursor: bool,
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
            xs = torch.stack([b[0] for b in batch]).to(device)
            ys = torch.stack([b[1] for b in batch]).to(device)
            yield xs, ys
            if max_batches is None and cursor == 0:
                break
        if use_cursor:
            self.eval_cursors[split] = cursor


# ============================================================
# 训练工具函数
# ============================================================
def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "norm" not in name.lower():
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def cosine_lr(step: int, total_steps: int, base_lr: float,
              warmup_steps: int, min_lr_scale: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)


class SwanLabMonitor:
    def __init__(self, backend: str, project: str, experiment_name: str, config: dict):
        self.enabled = backend == "swanlab" and swanlab is not None
        self.project = project
        self.experiment_name = experiment_name
        self.config = config

    def init(self):
        if not self.enabled:
            print("[swanlab] disabled")
            return
        try:
            api_key = os.environ.get("SWANLAB_API_KEY")
            if api_key:
                swanlab.login(api_key=api_key)
            swanlab.init(project=self.project, experiment_name=self.experiment_name, config=self.config)
            print(f"[swanlab] init: {self.project} / {self.experiment_name}")
        except Exception as exc:
            self.enabled = False
            print(f"[swanlab] init failed: {exc!r}")

    def log(self, metrics: dict, step: int):
        if self.enabled:
            try:
                swanlab.log(metrics, step=step)
            except Exception as exc:
                print(f"[swanlab] log failed: {exc!r}")

    def finish(self):
        if self.enabled:
            try:
                swanlab.finish()
            except Exception as exc:
                print(f"[swanlab] finish failed: {exc!r}")


@torch.no_grad()
def evaluate(model, data, split, batch_size, eval_batches, device, use_cursor):
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


def run_experiment(method_name: str, cfg: dict, data: LMData):
    device = torch.device(DEVICE)
    model = TinyDecoderLM(
        vocab_size=data.vocab_size,
        max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        attention_type=method_name,
        tie_weights=cfg["tie_weights"],
        use_pos_emb=cfg["use_pos_emb"],
    )
    if device.type == "cuda" and cfg["use_data_parallel"] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count / 1e6:.2f}M")

    optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
    history = []
    recent_losses = deque(maxlen=cfg["train_loss_window"])
    best_record, best_state = None, None
    started = time.perf_counter()

    monitor = SwanLabMonitor(
        backend=cfg["log_backend"],
        project=cfg["log_project"],
        experiment_name=f"{method_name}_{cfg['dataset_config']}_{cfg['d_model']}d_{cfg['num_layers']}l",
        config={**cfg, "method_name": method_name, "vocab_size": data.vocab_size, "model_params": param_count},
    )
    monitor.init()
    monitor.log({"model_params": param_count}, step=0)

    for step in range(1, cfg["steps"] + 1):
        lr = cosine_lr(step, cfg["steps"], cfg["lr"], cfg["warmup_steps"], cfg["min_lr_scale"])
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            inputs, labels = data.sample_train_batch(cfg["batch_size"], device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            (loss / cfg["grad_accum_steps"]).backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        train_loss = step_loss / cfg["grad_accum_steps"]
        recent_losses.append(train_loss)

        if step % cfg["eval_interval"] == 0 or step == 1 or step == cfg["steps"]:
            val = evaluate(model, data, "validation", cfg["batch_size"],
                           cfg["eval_batches"], device, use_cursor=True)
            elapsed = time.perf_counter() - started
            record = {
                "step": step, "lr": lr,
                "train_loss_raw": train_loss,
                "train_loss_avg": sum(recent_losses) / len(recent_losses),
                "val_loss": val["loss"], "val_ppl": val["perplexity"],
                "elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60,
            }
            history.append(record)
            if best_record is None or record["val_loss"] < best_record["val_loss"]:
                best_record = dict(record)
                src = model.module if isinstance(model, nn.DataParallel) else model
                best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
            monitor.log(record, step=step)
            print(
                f"[{method_name}] step={step:5d}  lr={lr:.2e}  "
                f"train={record['train_loss_avg']:.4f}  "
                f"val={record['val_loss']:.4f}  ppl={record['val_ppl']:.2f}  "
                f"time={record['elapsed_minutes']:.1f}min"
            )

    # 加载最佳 checkpoint
    src = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        src.load_state_dict(best_state)

    test = evaluate(model, data, "test", cfg["batch_size"],
                    cfg["final_test_batches"], device, use_cursor=False)

    summary = {
        "method": method_name, "model_params": param_count,
        "best_step":     best_record["step"]     if best_record else None,
        "best_val_loss": best_record["val_loss"]  if best_record else None,
        "best_val_ppl":  best_record["val_ppl"]   if best_record else None,
        "best_test_loss": test["loss"],
        "best_test_ppl":  test["perplexity"],
        "history": history, "config": cfg,
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_name}_{cfg['dataset_config']}_{cfg['d_model']}d_{cfg['num_layers']}l.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    monitor.log({"best_test_loss": summary["best_test_loss"],
                 "best_test_ppl":  summary["best_test_ppl"]}, step=cfg["steps"])
    monitor.finish()

    print("\n" + "=" * 60)
    print(f"🎉  实验完成: {method_name}")
    print(f"📦  参数量:         {param_count / 1e6:.2f} M")
    print(f"🏆  最佳 Val PPL:  {summary['best_val_ppl']:.2f}  (step {summary['best_step']})")
    print(f"🔥  最终 Test PPL: {summary['best_test_ppl']:.2f}")
    print(f"💾  结果已保存到:   {out_path}")
    print("=" * 60 + "\n")
    return summary


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # SWANLAB_API_KEY 可以在这里硬编码，也可以通过环境变量传入
    # os.environ["SWANLAB_API_KEY"] = "your_key_here"

    print("=" * 60)
    print("加载数据集（有缓存秒开，首次需要几分钟）...")
    print("=" * 60)
    data = LMData(LMCfg(
        data_source=           CFG["data_source"],
        local_data_dir=        CFG["local_data_dir"],
        dataset_name=          CFG["dataset_name"],
        dataset_config=        CFG["dataset_config"],
        text_field=            CFG["text_field"],
        tokenizer_name=        CFG["tokenizer_name"],
        max_seq_len=           CFG["max_seq_len"],
        add_eos_between_lines= CFG["add_eos_between_lines"],
    ))
    print(f"vocab_size = {data.vocab_size}")
    for split, ids in data.splits.items():
        print(f"  {split}: {ids.numel():,} tokens")

    results = {}
    for method_name in CFG["methods_to_run"]:
        print("\n" + "=" * 80)
        print(f"开始实验: {method_name}")
        print("=" * 80)
        results[method_name] = run_experiment(method_name, CFG, data)

    print("\n" + "=" * 80)
    print("所有实验结果汇总:")
    print("=" * 80)
    print(f"{'方法':<45} {'Val PPL':>10} {'Test PPL':>10} {'参数量':>10}")
    print("-" * 80)
    for name, s in sorted(results.items(), key=lambda x: x[1]["best_test_ppl"]):
        print(f"{name:<45} {s['best_val_ppl']:>10.2f} {s['best_test_ppl']:>10.2f} {s['model_params']/1e6:>9.2f}M")
