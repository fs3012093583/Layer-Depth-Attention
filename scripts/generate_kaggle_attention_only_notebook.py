import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = []

    cells.append(
        md_cell(
            """# Attention-Only Ablation Notebook

这个 notebook 用于在 **不修改残差结构和 FFN 结构** 的前提下，只比较注意力模块：

- `baseline`：标准 decoder-only Transformer
- `shared_kv_depth_memory`：层独立 `W_Q^{(l)}`，跨层共享 `W_K/W_V`，当前 token 额外读取“同位置跨层历史”的 attention 变体

特性：

- **单文件 notebook**
- **不依赖当前仓库中的任何自定义 Python 模块**
- **数据直接通过 Hugging Face `datasets` 加载**
- **SwanLab 记录默认写入新项目**
- **支持用户在配置区修改模型大小、数据集、训练步数**
- **支持多 GPU（Kaggle 若分配到两张 T4，会自动尝试 `DataParallel`）**

默认实现遵循本次实验约束：

1. 只修改 attention 模块  
2. residual / MLP / 训练配方在 baseline 和自定义方法之间保持一致  
3. 自定义方法采用：
   - 层独立 `W_Q^{(l)}`
   - 跨层共享 `W_K, W_V`
   - 历史 memory 为“同 token 位置的跨层历史”
4. 训练过程中验证使用**顺序游标评估**，避免每次只盯住验证集开头同一小段  
5. 最终测试默认使用**全量 test split**（可改成更大的固定批次数）
"""
        )
    )

    cells.append(
        code_cell(
            """# 安装依赖（Kaggle 首次运行时执行）
!pip -q install datasets transformers swanlab
"""
        )
    )

    cells.append(
        code_cell(
            """import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

try:
    import swanlab
except Exception:
    swanlab = None


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("device =", DEVICE, "num_gpus =", NUM_GPUS)
"""
        )
    )

    cells.append(
        code_cell(
            """# =========================
# 可改参数区
# =========================

CFG = {
    # 数据
    "data_source": "local_text",  # "local_text" or "hf"
    "local_data_dir": "/kaggle/input/wikitext2-raw",
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",   # 仅在 data_source="hf" 时使用
    "text_field": "text",
    "tokenizer_name": "gpt2",
    "add_eos_between_lines": True,

    # 模型
    "max_seq_len": 256,
    "d_model": 384,
    "num_layers": 8,
    "num_heads": 8,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "tie_weights": True,
    "use_pos_emb": True,

    # 训练
    "batch_size": 8,
    "grad_accum_steps": 1,
    "steps": 2000,
    "eval_interval": 200,
    "eval_batches": 20,            # 训练中验证采用顺序游标评估
    "final_test_batches": None,    # None 表示全量 test；也可改成整数
    "train_loss_window": 20,
    "lr": 3e-4,
    "min_lr_scale": 0.1,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "grad_clip": 1.0,

    # 运行
    "methods_to_run": ["baseline", "shared_kv_baseline", "shared_kv_depth_memory"],
    "use_data_parallel": True,
    "output_dir": "/kaggle/working/attention_only_ablation_outputs",

    # SwanLab（不要使用之前项目）
    "log_backend": "swanlab",  # "none" 或 "swanlab"
    "log_project": "Layer-Depth-Attention-Kaggle",
    "log_workspace": "justbook",
    "run_note": "attention_only_ablation_kaggle",
}

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
print(json.dumps(CFG, indent=2, ensure_ascii=False))
"""
        )
    )

    cells.append(
        code_cell(
            """@dataclass
class LMCfg:
    data_source: str
    local_data_dir: str
    dataset_name: str
    dataset_config: str
    text_field: str
    tokenizer_name: str
    max_seq_len: int
    add_eos_between_lines: bool = True


class LMData:
    def __init__(self, cfg: LMCfg):
        self.cfg = cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.eos_token_id
        self.dataset = None
        self.local_data_dir = Path(cfg.local_data_dir)
        if cfg.data_source == "hf":
            self.dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
        self.splits = self._tokenize_all_splits()
        self.eval_windows = {
            split: self._build_eval_windows(self.splits[split])
            for split in ["validation", "test"]
        }
        self.eval_cursors = {"validation": 0, "test": 0}

    def _join_text(self, split_name: str) -> str:
        if self.cfg.data_source == "local_text":
            text_path = self.local_data_dir / f"{split_name}.txt"
            text = text_path.read_text(encoding="utf-8")
        else:
            texts = self.dataset[split_name][self.cfg.text_field]
            text = "\\n".join(texts)
        if self.cfg.add_eos_between_lines:
            eos = self.tokenizer.eos_token
            text = text.replace("\\n", f" {eos} ")
        return text

    def _tokenize_split(self, split_name: str) -> torch.Tensor:
        text = self._join_text(split_name)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    def _tokenize_all_splits(self) -> Dict[str, torch.Tensor]:
        return {
            "train": self._tokenize_split("train"),
            "validation": self._tokenize_split("validation"),
            "test": self._tokenize_split("test"),
        }

    def _build_eval_windows(self, token_ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        chunk = self.cfg.max_seq_len + 1
        max_offset = token_ids.numel() - chunk
        starts = list(range(0, max_offset + 1, self.cfg.max_seq_len))
        windows = []
        for start in starts:
            piece = token_ids[start : start + chunk]
            windows.append((piece[:-1], piece[1:]))
        return windows

    def sample_train_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.splits["train"]
        chunk = self.cfg.max_seq_len + 1
        max_start = token_ids.numel() - chunk
        starts = torch.randint(0, max_start + 1, (batch_size,))
        windows = [token_ids[s : s + chunk] for s in starts.tolist()]
        batch = torch.stack(windows, dim=0).to(device)
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
            current = []
            for _ in range(batch_size):
                current.append(windows[cursor])
                cursor = (cursor + 1) % total
                if max_batches is None and cursor == 0 and len(current) > 0:
                    break
            xs = torch.stack([item[0] for item in current], dim=0).to(device)
            ys = torch.stack([item[1] for item in current], dim=0).to(device)
            yield xs, ys
            if max_batches is None and cursor == 0:
                break
        if use_cursor:
            self.eval_cursors[split] = cursor


data = LMData(
    LMCfg(
        data_source=CFG["data_source"],
        local_data_dir=CFG["local_data_dir"],
        dataset_name=CFG["dataset_name"],
        dataset_config=CFG["dataset_config"],
        text_field=CFG["text_field"],
        tokenizer_name=CFG["tokenizer_name"],
        max_seq_len=CFG["max_seq_len"],
        add_eos_between_lines=CFG["add_eos_between_lines"],
    )
)

print("vocab_size =", data.vocab_size)
for split_name, ids in data.splits.items():
    print(split_name, "tokens =", ids.numel())
"""
        )
    )

    cells.append(
        code_cell(
            """class MultiHeadAttentionBase(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class BaselineAttention(MultiHeadAttentionBase):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, x: torch.Tensor, past_kv=None):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self.out_proj(self.merge_heads(out)), (k, v)


class SharedKVProjector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)


class SharedKVBaselineAttention(MultiHeadAttentionBase):
    '''
    只做共享 KV 的 baseline：
    - 每层独立 q_proj
    - 跨层共享 k_proj / v_proj
    - 不使用跨层 memory
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.shared_kv = shared_kv

    def forward(self, x: torch.Tensor, past_kv=None):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.shared_kv.k_proj(x))
        v = self.split_heads(self.shared_kv.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self.out_proj(self.merge_heads(out)), (k, v)


class SharedKVDepthMemoryAttention(MultiHeadAttentionBase):
    '''
    只修改 attention：
    - 每层独立 q_proj
    - 跨层共享 k_proj / v_proj
    - 历史 memory 为“同 token 位置的跨层历史”
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.shared_kv = shared_kv

    def forward(self, x: torch.Tensor, past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.shared_kv.k_proj(x))
        v = self.split_heads(self.shared_kv.v_proj(x))

        seq_len = x.size(1)
        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(self.causal_mask(seq_len, x.device), float('-inf'))

        if past_kv:
            past_keys = torch.stack([item[0] for item in past_kv], dim=3)      # [B,H,S,Lpast,d]
            past_values = torch.stack([item[1] for item in past_kv], dim=3)    # [B,H,S,Lpast,d]
            memory_scores = (q.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))
            token_w = weights[..., :seq_len]
            mem_w = weights[..., seq_len:]
            token_ctx = torch.matmul(token_w, v)
            mem_ctx = (mem_w.unsqueeze(-1) * past_values).sum(dim=3)
            out = token_ctx + mem_ctx
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            out = torch.matmul(weights, v)

        return self.out_proj(self.merge_heads(out)), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int, dropout: float, attention: nn.Module):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = attention
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, past_kv=None):
        attn_out, current_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, current_kv


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        attention_type: str,
        tie_weights: bool = True,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.use_pos_emb = use_pos_emb
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        shared_kv = SharedKVProjector(d_model) if attention_type in ["shared_kv_baseline", "shared_kv_depth_memory"] else None
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == "baseline":
                attention = BaselineAttention(d_model, num_heads, dropout)
            elif attention_type == "shared_kv_baseline":
                attention = SharedKVBaselineAttention(d_model, num_heads, dropout, shared_kv)
            elif attention_type == "shared_kv_depth_memory":
                attention = SharedKVDepthMemoryAttention(d_model, num_heads, dropout, shared_kv)
            else:
                raise ValueError(attention_type)
            self.blocks.append(TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attention))

        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        x = self.token_emb(input_ids)
        if self.use_pos_emb:
            pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
            x = x + self.pos_emb(pos)
        x = self.dropout(x)

        past_kv = []
        for block in self.blocks:
            if self.attention_type == "shared_kv_depth_memory":
                x, current_kv = block(x, past_kv=past_kv)
                past_kv.append(current_kv)
            else:
                x, _ = block(x, past_kv=None)
        x = self.final_norm(x)
        return self.lm_head(x)
"""
        )
    )

    cells.append(
        code_cell(
            """def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "norm" not in name.lower():
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


class SwanLabMonitor:
    def __init__(self, backend: str, project: str, experiment_name: str, config: dict):
        self.backend = backend
        self.project = project
        self.experiment_name = experiment_name
        self.config = config
        self.enabled = backend == "swanlab" and swanlab is not None

    def init(self):
        if not self.enabled:
            print("[swanlab] disabled")
            return
        try:
            api_key = os.environ.get("SWANLAB_API_KEY")
            if api_key:
                swanlab.login(api_key=api_key)
            else:
                swanlab.login()
            swanlab.init(project=self.project, experiment_name=self.experiment_name, config=self.config)
            print(f"[swanlab] init succeeded: project={self.project} experiment={self.experiment_name}")
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
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    for inputs, labels in data.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        max_batches=eval_batches,
        use_cursor=use_cursor,
    ):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        total_batches += 1
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "batches": total_batches,
    }


def run_experiment(method_name: str, cfg: dict):
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

    optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
    history = []
    recent_train_losses = deque(maxlen=cfg["train_loss_window"])
    best_record = None
    best_state = None
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
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        step_loss_sum = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            inputs, labels = data.sample_train_batch(cfg["batch_size"], device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            (loss / cfg["grad_accum_steps"]).backward()
            step_loss_sum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        train_loss = step_loss_sum / cfg["grad_accum_steps"]
        recent_train_losses.append(train_loss)

        if step % cfg["eval_interval"] == 0 or step == 1 or step == cfg["steps"]:
            val_metrics = evaluate(
                model=model,
                data=data,
                split="validation",
                batch_size=cfg["batch_size"],
                eval_batches=cfg["eval_batches"],
                device=device,
                use_cursor=True,
            )
            elapsed_seconds = time.perf_counter() - started
            record = {
                "step": step,
                "lr": lr,
                "train_loss_raw": train_loss,
                "train_loss_avg": sum(recent_train_losses) / len(recent_train_losses),
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["perplexity"],
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
            }
            history.append(record)
            if best_record is None or record["val_loss"] < best_record["val_loss"]:
                best_record = dict(record)
                source_model = model.module if isinstance(model, nn.DataParallel) else model
                best_state = {k: v.detach().cpu().clone() for k, v in source_model.state_dict().items()}
            monitor.log(record, step=step)
            print(
                f"method={method_name} step={step} lr={lr:.6f} "
                f"train_loss_avg={record['train_loss_avg']:.4f} "
                f"val_loss={record['val_loss']:.4f} val_ppl={record['val_ppl']:.2f} "
                f"elapsed_min={record['elapsed_minutes']:.2f}"
            )

    source_model = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        source_model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        data=data,
        split="test",
        batch_size=cfg["batch_size"],
        eval_batches=cfg["final_test_batches"],
        device=device,
        use_cursor=False,
    )

    summary = {
        "method": method_name,
        "model_params": param_count,
        "best_step": None if best_record is None else best_record["step"],
        "best_val_loss": None if best_record is None else best_record["val_loss"],
        "best_val_ppl": None if best_record is None else best_record["val_ppl"],
        "best_test_loss": test_metrics["loss"],
        "best_test_ppl": test_metrics["perplexity"],
        "history": history,
        "config": cfg,
    }

    out_path = Path(cfg["output_dir"]) / f"{method_name}_{cfg['dataset_config']}_{cfg['d_model']}d_{cfg['num_layers']}l.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    monitor.log({"best_test_loss": summary["best_test_loss"], "best_test_ppl": summary["best_test_ppl"]}, step=cfg["steps"])
    monitor.finish()
    return summary
"""
        )
    )

    cells.append(
        code_cell(
            """results = {}
for method_name in CFG["methods_to_run"]:
    print("=" * 80)
    print("running", method_name)
    print("=" * 80)
    results[method_name] = run_experiment(method_name, CFG)

results
"""
        )
    )

    cells.append(
        code_cell(
            """import pandas as pd

rows = []
for name, summary in results.items():
    rows.append({
        "method": name,
        "model_params": summary["model_params"],
        "best_step": summary["best_step"],
        "best_val_loss": summary["best_val_loss"],
        "best_val_ppl": summary["best_val_ppl"],
        "best_test_loss": summary["best_test_loss"],
        "best_test_ppl": summary["best_test_ppl"],
    })

df = pd.DataFrame(rows).sort_values("best_test_ppl")
df
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kaggle_attention_only_ablation.ipynb"
    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
