"""
Train CIFAR100 baselines and layer-depth attention variants from a single script.

This script intentionally does not reuse the old vision training stack. Instead it:
- uses the new ViT-style models defined in `ablation_models.py`
- adds a clean CIFAR-style ResNet baseline for architecture comparison
- keeps the CLI close to the language ablation script, so experiments are easier to track
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from layer_depth_attention.ablation_models import TinyVisionTransformerAblation

try:
    import swanlab
except Exception:
    swanlab = None


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
VIT_METHODS = {
    "baseline",
    "shared_kv_depth_memory_dualq",
    "shared_kv_depth_memory_dualq_sublayer",
    "depth_memory_reuse_row_qkv",
    "depth_memory_hidden_states_sublayer",
}
RESNET_METHODS = {"resnet18_cifar"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR100 baselines and vision ablations.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "shared_kv_depth_memory_dualq_sublayer", "resnet18_cifar"],
        help="ViT attention variants or resnet18_cifar",
    )
    parser.add_argument("--dataset-name", default="cifar100")
    parser.add_argument("--data-source", choices=["hf", "local_dir"], default="hf")
    parser.add_argument("--local-data-dir", default="")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "cifar100_ablation"))

    parser.add_argument("--vit-d-model", type=int, default=256)
    parser.add_argument("--vit-num-layers", type=int, default=6)
    parser.add_argument("--vit-num-heads", type=int, default=8)
    parser.add_argument("--vit-mlp-ratio", type=int, default=4)
    parser.add_argument("--vit-patch-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vit-lr", type=float, default=3e-4)
    parser.add_argument("--resnet-lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--save-checkpoints", choices=["on", "off"], default="on")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-backend", choices=["none", "swanlab"], default="swanlab")
    parser.add_argument("--log-project", default="Layer-Depth-Attention-CV-Ablation")
    parser.add_argument("--log-workspace", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_bool(flag: str) -> bool:
    return flag == "on"


class TrainTransform:
    """Minimal CIFAR-style augmentation without depending on torchvision."""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = ImageOps.expand(image, border=4, fill=0)
        max_off = image.size[0] - self.image_size
        left = random.randint(0, max_off)
        top = random.randint(0, max_off)
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return pil_to_normalized_tensor(image)


class EvalTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return pil_to_normalized_tensor(image)


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(CIFAR100_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[index]
        image = item["img"]
        label = int(item["fine_label"])
        return self.transform(image), label


class LocalCifar100Dataset(Dataset):
    """Read the official CIFAR100 python-format files from a local directory."""

    def __init__(self, records: List[Tuple[np.ndarray, int]], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_array, label = self.records[index]
        image = Image.fromarray(image_array, mode="RGB")
        return self.transform(image), label


def load_local_cifar100_records(root: Path, split_name: str) -> List[Tuple[np.ndarray, int]]:
    split_path = root / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Missing CIFAR100 split file: {split_path}")
    with split_path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    raw = payload["data"]
    labels = payload["fine_labels"]
    images = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return [(images[i], int(labels[i])) for i in range(len(labels))]


def build_cifar100_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.data_source == "local_dir":
        if not args.local_data_dir:
            raise ValueError("--local-data-dir is required when --data-source local_dir")
        root = Path(args.local_data_dir)
        train_records = load_local_cifar100_records(root, "train")
        test_records = load_local_cifar100_records(root, "test")

        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(train_records))
        rng.shuffle(indices)
        val_size = int(len(train_records) * args.val_ratio)
        val_idx = set(indices[:val_size].tolist())
        train_subset = [train_records[i] for i in range(len(train_records)) if i not in val_idx]
        val_subset = [train_records[i] for i in range(len(train_records)) if i in val_idx]

        train_size = len(train_subset)
        train_ds = LocalCifar100Dataset(train_subset, TrainTransform(args.image_size))
        val_ds = LocalCifar100Dataset(val_subset, EvalTransform(args.image_size))
        test_ds = LocalCifar100Dataset(test_records, EvalTransform(args.image_size))
        test_size = len(test_records)
    else:
        dataset = load_dataset(args.dataset_name)
        train_full = dataset["train"]
        test_set = dataset["test"]

        val_size = int(len(train_full) * args.val_ratio)
        train_size = len(train_full) - val_size
        split = train_full.train_test_split(test_size=val_size, seed=args.seed)
        train_set = split["train"]
        val_set = split["test"]

        train_ds = HFDatasetWrapper(train_set, TrainTransform(args.image_size))
        val_ds = HFDatasetWrapper(val_set, EvalTransform(args.image_size))
        test_ds = HFDatasetWrapper(test_set, EvalTransform(args.image_size))
        test_size = len(test_set)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    print(f"[data] source={args.data_source} train={train_size} val={val_size} test={test_size}")
    return train_loader, val_loader, test_loader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class CifarResNet(nn.Module):
    """CIFAR-style ResNet18 baseline without torchvision dependency."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_model(method: str, args: argparse.Namespace) -> nn.Module:
    if method in VIT_METHODS:
        return TinyVisionTransformerAblation(
            image_size=args.image_size,
            patch_size=args.vit_patch_size,
            num_classes=100,
            d_model=args.vit_d_model,
            num_layers=args.vit_num_layers,
            num_heads=args.vit_num_heads,
            mlp_ratio=args.vit_mlp_ratio,
            dropout=args.dropout,
            attention_type=method,
        )
    if method == "resnet18_cifar":
        return CifarResNet(num_classes=100)
    raise ValueError(f"Unsupported method: {method}")


def build_optimizer(method: str, model: nn.Module, args: argparse.Namespace):
    if method == "resnet18_cifar":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.resnet_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.vit_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


@dataclass
class EvalStats:
    loss: float
    acc: float


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
            kwargs = {
                "project": self.project,
                "experiment_name": self.experiment_name,
                "config": self.config,
            }
            api_key = None
            try:
                import os

                api_key = os.environ.get("SWANLAB_API_KEY")
            except Exception:
                api_key = None
            if api_key:
                swanlab.login(api_key=api_key)
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += images.size(0)
    return EvalStats(loss=total_loss / total_examples, acc=total_correct / total_examples)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_experiment_name(method: str, args: argparse.Namespace) -> str:
    backbone = "resnet18" if method == "resnet18_cifar" else f"vit{args.vit_d_model}d_{args.vit_num_layers}l"
    return f"{method}_cifar100_{backbone}_ep{args.epochs}_seed{args.seed}"


def build_monitor_config(method: str, args: argparse.Namespace, model_params: int) -> Dict[str, object]:
    return {
        "method": method,
        "dataset_name": args.dataset_name,
        "data_source": args.data_source,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "vit_d_model": args.vit_d_model,
        "vit_num_layers": args.vit_num_layers,
        "vit_num_heads": args.vit_num_heads,
        "vit_patch_size": args.vit_patch_size,
        "dropout": args.dropout,
        "vit_lr": args.vit_lr,
        "resnet_lr": args.resnet_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "model_params": model_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path,
    method: str,
    epoch: int,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{method}_epoch{epoch:03d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
        },
        ckpt_path,
    )
    print(f"[checkpoint] saved epoch={epoch} -> {ckpt_path}")
    return ckpt_path


def save_model_state(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def train_one_method(
    method: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_root: Path,
) -> Dict[str, object]:
    method_dir = run_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(method, args).to(device)
    param_count = count_parameters(model)
    optimizer, scheduler = build_optimizer(method, model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    monitor = SwanLabMonitor(
        backend=args.log_backend,
        project=args.log_project,
        experiment_name=build_experiment_name(method, args),
        config=build_monitor_config(method, args, param_count),
        workspace=args.log_workspace,
    )
    monitor.init()
    monitor.log({"model_params": param_count}, step=0)

    best_val_acc = -1.0
    best_val_loss = math.inf
    best_epoch = -1
    checkpoint_paths: List[str] = []
    best_state_dict = None
    should_save = as_bool(args.save_checkpoints)

    print(f"[train] method={method} params={param_count/1e6:.2f}M")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += images.size(0)

        scheduler.step()
        epoch_time = time.time() - start
        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples
        print(
            f"[epoch {epoch:03d}] method={method} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} time={epoch_time:.1f}s"
        )
        monitor.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "epoch_time_sec": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_stats = evaluate(model, val_loader, device, criterion)
            print(
                f"[eval {epoch:03d}] method={method} val_loss={val_stats.loss:.4f} "
                f"val_acc={val_stats.acc:.4f}"
            )
            monitor.log({"val_loss": val_stats.loss, "val_acc": val_stats.acc}, step=epoch)
            if val_stats.acc > best_val_acc:
                best_val_acc = val_stats.acc
                best_val_loss = val_stats.loss
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if should_save:
                    best_path = method_dir / "best.pt"
                    save_model_state(model, best_path)
                    print(f"[checkpoint] updated best -> {best_path}")

        if should_save and args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            checkpoint_paths.append(str(save_checkpoint(model, optimizer, scheduler, method_dir, method, epoch)))

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_stats = evaluate(model, test_loader, device, criterion)
    print(f"[test] method={method} test_loss={test_stats.loss:.4f} test_acc={test_stats.acc:.4f}")
    monitor.log(
        {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_test_loss": test_stats.loss,
            "final_test_acc": test_stats.acc,
        },
        step=args.epochs,
    )

    if should_save:
        final_path = save_checkpoint(model, optimizer, scheduler, method_dir, method, args.epochs)
        checkpoint_paths.append(str(final_path))

    result = {
        "method": method,
        "params": count_parameters(model),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_test_acc": test_stats.acc,
        "final_test_loss": test_stats.loss,
        "save_checkpoints": should_save,
        "checkpoints": checkpoint_paths,
    }
    with (method_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    monitor.finish()
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader, test_loader = build_cifar100_dataloaders(args)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cifar100_"
        f"vit{args.vit_d_model}d_{args.vit_num_layers}l_ep{args.epochs}_seed{args.seed}"
    )
    run_root = Path(args.output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    with (run_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    all_results = []
    for method in args.methods:
        result = train_one_method(method, args, train_loader, val_loader, test_loader, device, run_root)
        all_results.append(result)

    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
