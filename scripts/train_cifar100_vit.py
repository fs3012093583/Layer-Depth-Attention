import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from layer_depth_attention.vision_model import TinyVisionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="D:/Projects/data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--attention-type",
        choices=["baseline", "depth_memory_value_reproj_normed"],
        default="baseline",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    data_root = str(Path(args.data_root))
    train_set = datasets.CIFAR100(root=data_root, train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_root, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: TinyVisionTransformer, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += labels.size(0)
    model.train()
    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_loaders(args)
    model = TinyVisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=100,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_type=args.attention_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=-1) == labels).sum().item()
            running_examples += labels.size(0)

        test_metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_examples, 1),
            "train_acc": running_correct / max(running_examples, 1),
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} "
            f"test_loss={record['test_loss']:.4f} test_acc={record['test_acc']:.4f}"
        )

    summary = {"history": history, "config": vars(args)}
    output_name = args.output or f"artifacts/cifar100_{args.attention_type}.json"
    output_path = ROOT / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved metrics to {output_path}")


if __name__ == "__main__":
    main()
