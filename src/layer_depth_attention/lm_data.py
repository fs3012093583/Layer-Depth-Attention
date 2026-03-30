from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
import warnings

import torch
from transformers import GPT2Tokenizer


@dataclass
class WikiTextLMConfig:
    data_dir: Path
    tokenizer_dir: Path
    seq_len: int
    add_eos_between_lines: bool = True


class WikiTextLanguageModelingData:
    def __init__(self, config: WikiTextLMConfig) -> None:
        self.config = config
        self.tokenizer = self._build_tokenizer(config.tokenizer_dir)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.encoder["<|endoftext|>"]
        self.splits = self._load_splits()

    def _build_tokenizer(self, tokenizer_dir: Path) -> GPT2Tokenizer:
        vocab_file = tokenizer_dir / "vocab.json"
        merges_file = tokenizer_dir / "merges.txt"
        tokenizer = GPT2Tokenizer(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file),
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_split(self, split: str) -> torch.Tensor:
        cache_path = self.config.data_dir / f"{split}_gpt2_ids.pt"
        if cache_path.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using `torch.load` with `weights_only=False`",
                    category=FutureWarning,
                )
                return torch.load(cache_path)
        text_path = self.config.data_dir / f"{split}.txt"
        text = text_path.read_text(encoding="utf-8")
        if self.config.add_eos_between_lines:
            text = text.replace("\n", f" {self.tokenizer.eos_token} ")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tensor = torch.tensor(token_ids, dtype=torch.long)
        torch.save(tensor, cache_path)
        return tensor

    def _load_splits(self) -> Dict[str, torch.Tensor]:
        return {
            split: self._load_split(split)
            for split in ("train", "validation", "test")
        }

    def sample_train_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        train_ids = self.splits["train"]
        max_start = train_ids.numel() - (self.config.seq_len + 1)
        starts = torch.randint(0, max_start + 1, (batch_size,))
        windows = [train_ids[start : start + self.config.seq_len + 1] for start in starts.tolist()]
        batch = torch.stack(windows, dim=0).to(device)
        return batch[:, :-1], batch[:, 1:]

    def iter_eval_batches(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        token_ids = self.splits[split]
        chunk_size = self.config.seq_len + 1
        max_offset = token_ids.numel() - chunk_size
        starts = range(0, max_offset + 1, self.config.seq_len)
        batch_inputs = []
        batch_labels = []
        yielded = 0
        for start in starts:
            window = token_ids[start : start + chunk_size]
            batch_inputs.append(window[:-1])
            batch_labels.append(window[1:])
            if len(batch_inputs) == batch_size:
                yielded += 1
                yield torch.stack(batch_inputs, dim=0).to(device), torch.stack(batch_labels, dim=0).to(device)
                batch_inputs = []
                batch_labels = []
                if max_batches is not None and yielded >= max_batches:
                    return
        if batch_inputs and (max_batches is None or yielded < max_batches):
            yield torch.stack(batch_inputs, dim=0).to(device), torch.stack(batch_labels, dim=0).to(device)
