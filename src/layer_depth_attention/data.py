from dataclasses import dataclass
import random
from typing import Tuple

import torch


@dataclass
class AssocRecallConfig:
    vocab_size: int = 64
    num_pairs: int = 6
    pad_token: int = 0
    bos_token: int = 1
    sep_token: int = 2

    @property
    def sequence_length(self) -> int:
        # [BOS] + 2 * num_pairs + [SEP] + query + answer
        return 2 * self.num_pairs + 4


class AssociativeRecallDataset:
    def __init__(self, config: AssocRecallConfig, seed: int = 0) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.key_start = 3
        self.value_start = self.key_start + config.num_pairs

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        inputs = torch.full(
            (batch_size, cfg.sequence_length - 1),
            cfg.pad_token,
            dtype=torch.long,
        )
        labels = torch.full_like(inputs, -100)

        for row in range(batch_size):
            keys = list(range(self.key_start, self.key_start + cfg.num_pairs))
            values = list(range(self.value_start, self.value_start + cfg.num_pairs))
            self.rng.shuffle(values)

            query_index = self.rng.randrange(cfg.num_pairs)
            query_key = keys[query_index]
            answer_value = values[query_index]

            full_sequence = [cfg.bos_token]
            for key, value in zip(keys, values):
                full_sequence.extend([key, value])
            full_sequence.extend([cfg.sep_token, query_key, answer_value])

            token_tensor = torch.tensor(full_sequence, dtype=torch.long)
            inputs[row] = token_tensor[:-1]
            labels[row, -1] = token_tensor[-1]

        return inputs, labels
