import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


KVCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FFNCache = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttentionBase(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _split_qkv(self, x: torch.Tensor) -> KVCache:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        return reshape_heads(q), reshape_heads(k), reshape_heads(v)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def _split_projected(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _kv_proj(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_weight = self.qkv_proj.weight[self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim]
        v_weight = self.qkv_proj.weight[2 * self.num_heads * self.head_dim :]
        k_bias = self.qkv_proj.bias[self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim]
        v_bias = self.qkv_proj.bias[2 * self.num_heads * self.head_dim :]
        k = torch.nn.functional.linear(x, k_weight, k_bias)
        v = torch.nn.functional.linear(x, v_weight, v_bias)
        return self._split_projected(k), self._split_projected(v)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask


class CausalSelfAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        del past_kv
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class DualAxisMemoryAttention(MultiHeadAttentionBase):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__(d_model, num_heads, dropout)
        self.column_q_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
        past_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        del past_kv
        q_row, k, v = self._split_qkv(x)
        q_col = self._split_projected(self.column_q_proj(x))
        seq_len = x.size(1)

        token_scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(self._causal_mask(seq_len, x.device), float("-inf"))

        if past_states:
            memory_bank = torch.stack(past_states, dim=2)
            memory_bank = F.layer_norm(memory_bank, (memory_bank.size(-1),))
            batch_size, memory_seq_len, num_past_layers, _ = memory_bank.shape
            memory_bank = memory_bank.view(
                batch_size,
                memory_seq_len,
                num_past_layers,
                self.num_heads,
                self.head_dim,
            ).permute(0, 3, 1, 2, 4)

            memory_scores = (q_col.unsqueeze(3) * memory_bank).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * memory_bank).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q_row)


class LayerDepthMemoryAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        batch_size, _, seq_len, _ = q.shape

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_keys = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            memory_scores = (q.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * past_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class LayerDepthValueReprojAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            num_past_layers = past_values.size(3)
            past_values_full = (
                past_values.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            reproj_inputs = past_values_full.view(x.size(0), seq_len * num_past_layers, -1)
            reproj_keys, reproj_values = self._kv_proj(reproj_inputs)
            reproj_keys = reproj_keys.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)
            reproj_values = reproj_values.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)

            memory_scores = (q.unsqueeze(3) * reproj_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * reproj_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class LayerDepthValueReprojNormedAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            num_past_layers = past_values.size(3)
            past_values_full = (
                past_values.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            reproj_inputs = past_values_full.view(x.size(0), seq_len * num_past_layers, -1)
            reproj_inputs = F.layer_norm(reproj_inputs, (reproj_inputs.size(-1),))
            reproj_keys, reproj_values = self._kv_proj(reproj_inputs)
            reproj_keys = reproj_keys.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)
            reproj_values = reproj_values.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)

            memory_scores = (q.unsqueeze(3) * reproj_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * reproj_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class LayerDepthValueReprojDualQAttention(MultiHeadAttentionBase):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)
        self.column_q_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q_row, k, v = self._split_qkv(x)
        q_col = self._split_projected(self.column_q_proj(x))
        seq_len = x.size(1)

        token_scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_keys = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            memory_scores = (q_col.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * past_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q_row)


class LayerDepthValueReprojNormedDualQAttention(MultiHeadAttentionBase):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)
        self.column_q_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q_row, k, v = self._split_qkv(x)
        q_col = self._split_projected(self.column_q_proj(x))
        seq_len = x.size(1)

        token_scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            num_past_layers = past_values.size(3)
            past_values_full = (
                past_values.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            reproj_inputs = past_values_full.view(x.size(0), seq_len * num_past_layers, -1)
            reproj_inputs = F.layer_norm(reproj_inputs, (reproj_inputs.size(-1),))
            reproj_keys, reproj_values = self._kv_proj(reproj_inputs)
            reproj_keys = reproj_keys.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)
            reproj_values = reproj_values.view(x.size(0), self.num_heads, seq_len, num_past_layers, self.head_dim)

            memory_scores = (q_col.unsqueeze(3) * reproj_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * reproj_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q_row)


class LayerDepthQKVReprojAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            past_keys = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            past_queries = torch.stack([item[2] for item in past_kv], dim=3)
            num_past_layers = past_keys.size(3)

            past_queries_full = (
                past_queries.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            past_keys_full = (
                past_keys.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            past_values_full = (
                past_values.permute(0, 2, 3, 1, 4)
                .contiguous()
                .view(x.size(0), seq_len, num_past_layers, -1)
            )
            memory_inputs = torch.cat(
                [past_queries_full, past_keys_full, past_values_full],
                dim=2,
            )
            num_memory_slots = memory_inputs.size(2)
            memory_inputs = memory_inputs.view(x.size(0), seq_len * num_memory_slots, -1)
            memory_inputs = F.layer_norm(memory_inputs, (memory_inputs.size(-1),))
            reproj_keys, reproj_values = self._kv_proj(memory_inputs)
            reproj_keys = reproj_keys.view(x.size(0), self.num_heads, seq_len, num_memory_slots, self.head_dim)
            reproj_values = reproj_values.view(x.size(0), self.num_heads, seq_len, num_memory_slots, self.head_dim)

            memory_scores = (q.unsqueeze(3) * reproj_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * reproj_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class LayerDepth2DPrefixAttention(MultiHeadAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)
        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            batch_size = x.size(0)
            num_past_layers = len(past_kv)
            past_keys = torch.stack([item[0] for item in past_kv], dim=2)
            past_values = torch.stack([item[1] for item in past_kv], dim=2)
            past_keys = past_keys.reshape(batch_size, self.num_heads, num_past_layers * seq_len, self.head_dim)
            past_values = past_values.reshape(batch_size, self.num_heads, num_past_layers * seq_len, self.head_dim)

            memory_scores = torch.matmul(q, past_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            prefix_positions = torch.arange(seq_len, device=x.device).repeat(num_past_layers)
            query_positions = torch.arange(seq_len, device=x.device)
            memory_mask = prefix_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            memory_scores = memory_scores.masked_fill(memory_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            all_scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(all_scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, v)
            memory_context = torch.matmul(memory_weights, past_values)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class Top1MoE(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_probs, top_indices = router_probs.max(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, 1, x.size(-1))
        chosen = expert_outputs.gather(2, gather_index).squeeze(2)
        return self.dropout(chosen * top_probs)



# 代替全连接和前馈神经网络
class FeedForwardQAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        x: torch.Tensor,
        past_ffn: Optional[List[FFNCache]] = None,
    ) -> Tuple[torch.Tensor, FFNCache]:
        q = self._split_heads(self.q_proj(x))
        values = self._split_heads(x)
        seq_len = x.size(1)
        token_scores = torch.matmul(q, q.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_ffn:
            past_queries = torch.stack([item[0] for item in past_ffn], dim=3)
            past_values = torch.stack([item[1] for item in past_ffn], dim=3)
            memory_scores = (q.unsqueeze(3) * past_queries).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, values)
            memory_context = (memory_weights.unsqueeze(-1) * past_values).sum(dim=3)
            context = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            context = torch.matmul(weights, values)

        output = self.activation(self.out_proj(self._merge_heads(context)))
        return output, (q, values)


class FeedForwardDualQAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.row_q_proj = nn.Linear(d_model, d_model)
        self.col_q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        x: torch.Tensor,
        past_ffn: Optional[List[FFNCache]] = None,
    ) -> Tuple[torch.Tensor, FFNCache]:
        q_row = self._split_heads(self.row_q_proj(x))
        q_col = self._split_heads(self.col_q_proj(x))
        values = self._split_heads(x)
        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, q_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._causal_mask(seq_len, x.device)
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_ffn:
            past_queries = torch.stack([item[0] for item in past_ffn], dim=3)
            past_values = torch.stack([item[1] for item in past_ffn], dim=3)
            memory_scores = (q_col.unsqueeze(3) * past_queries).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., :seq_len]
            memory_weights = weights[..., seq_len:]
            token_context = torch.matmul(token_weights, values)
            memory_context = (memory_weights.unsqueeze(-1) * past_values).sum(dim=3)
            context = token_context + memory_context
        else:
            weights = torch.softmax(token_scores, dim=-1)
            weights = self.dropout(weights)
            context = torch.matmul(weights, values)

        output = self.activation(self.out_proj(self._merge_heads(context)))
        return output, (q_col, values)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        attention_type: str,
        ffn_type: str,
        num_experts: int,
        attn_residual: bool,
        ffn_residual: bool,
    ) -> None:
        super().__init__()
        self.attn_residual = attn_residual
        self.ffn_residual = ffn_residual
        self.attn_norm = nn.LayerNorm(d_model)
        if attention_type == "baseline":
            self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        elif attention_type == "dual_axis_memory":
            self.attn = DualAxisMemoryAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory":
            self.attn = LayerDepthMemoryAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj":
            self.attn = LayerDepthValueReprojAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj_normed":
            self.attn = LayerDepthValueReprojNormedAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj_dualq":
            self.attn = LayerDepthValueReprojDualQAttention(
                d_model,
                num_heads,
                dropout,
            )
        elif attention_type == "depth_memory_value_reproj_normed_dualq":
            self.attn = LayerDepthValueReprojNormedDualQAttention(
                d_model,
                num_heads,
                dropout,
            )
        elif attention_type == "depth_memory_qkv_reproj":
            self.attn = LayerDepthQKVReprojAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_2d_prefix":
            self.attn = LayerDepth2DPrefixAttention(d_model, num_heads, dropout)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
        self.mlp_norm = nn.LayerNorm(d_model)
        if ffn_type == "dense":
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_ratio * d_model),
                nn.GELU(),
                nn.Linear(mlp_ratio * d_model, d_model),
                nn.Dropout(dropout),
            )
        elif ffn_type == "moe":
            self.mlp = Top1MoE(d_model, mlp_ratio * d_model, num_experts, dropout)
        elif ffn_type == "q_attn":
            self.mlp = FeedForwardQAttention(d_model, num_heads, dropout)
        elif ffn_type == "q_attn_dualq":
            self.mlp = FeedForwardDualQAttention(d_model, num_heads, dropout)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
        past_ffn: Optional[List[FFNCache]] = None,
        past_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, KVCache, Optional[FFNCache]]:
        if isinstance(self.attn, DualAxisMemoryAttention):
            attn_out, current_kv = self.attn(self.attn_norm(x), past_kv=past_kv, past_states=past_states)
        else:
            attn_out, current_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        if self.attn_residual:
            x = x + attn_out
        else:
            x = attn_out
        if isinstance(self.mlp, (FeedForwardQAttention, FeedForwardDualQAttention)):
            mlp_out, current_ffn = self.mlp(self.mlp_norm(x), past_ffn=past_ffn)
        else:
            mlp_out = self.mlp(self.mlp_norm(x))
            current_ffn = None
        if self.ffn_residual:
            x = x + mlp_out
        else:
            x = mlp_out
        return x, current_kv, current_ffn


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_type: str = "baseline",
        num_experts: int = 4,
        attn_residual: bool = True,
        ffn_residual: bool = True,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        residual_attention_types = {
            "attn_residuals",
            "attn_residuals_dual_axis",
            "dual_axis_full",
            "attn_residuals_value_reproj",
            "attn_residuals_value_reproj_normed",
            "attn_residuals_moe",
        }
        if attention_type == "attn_residuals":
            block_attention_type = "baseline"
            block_ffn_type = "dense"
        elif attention_type == "attn_residuals_dual_axis":
            block_attention_type = "baseline"
            block_ffn_type = "dense"
        elif attention_type == "dual_axis_full":
            block_attention_type = "dual_axis_memory"
            block_ffn_type = "dense"
        elif attention_type == "attn_residuals_value_reproj":
            block_attention_type = "depth_memory_value_reproj"
            block_ffn_type = "dense"
        elif attention_type == "attn_residuals_value_reproj_normed":
            block_attention_type = "depth_memory_value_reproj_normed"
            block_ffn_type = "dense"
        elif attention_type == "attn_residuals_moe":
            block_attention_type = "baseline"
            block_ffn_type = "moe"
        elif attention_type == "depth_memory_value_reproj_normed_ffn_qattn":
            block_attention_type = "depth_memory_value_reproj_normed"
            block_ffn_type = "q_attn"
        elif attention_type == "depth_memory_value_reproj_normed_dualq_ffn_qattn_dualq":
            block_attention_type = "depth_memory_value_reproj_normed_dualq"
            block_ffn_type = "q_attn_dualq"
        else:
            block_attention_type = attention_type
            block_ffn_type = "dense"
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_type=block_attention_type,
                    ffn_type=block_ffn_type,
                    num_experts=num_experts,
                    attn_residual=attn_residual,
                    ffn_residual=ffn_residual,
                )
                for _ in range(num_layers)
            ]
        )
        if attention_type in residual_attention_types:
            self.attn_res_queries = nn.Parameter(torch.empty(num_layers, d_model))
            self.mlp_res_queries = nn.Parameter(torch.empty(num_layers, d_model))
            self.final_res_query = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(self.attn_res_queries, std=0.02)
            nn.init.normal_(self.mlp_res_queries, std=0.02)
            nn.init.normal_(self.final_res_query, std=0.02)
        if attention_type in {"attn_residuals_dual_axis", "dual_axis_full"}:
            self.attn_res_row_q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.mlp_res_row_q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.final_res_row_q_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def _rms_norm_tensor(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _attn_res_mix(
        self,
        embedding: torch.Tensor,
        history: List[torch.Tensor],
        query: torch.Tensor,
    ) -> torch.Tensor:
        values = [embedding] + history
        stacked = torch.stack(values, dim=2)
        normed = self._rms_norm_tensor(stacked)
        # Scale the depth scores like standard dot-product attention so deeper/wider
        # models do not make the residual router overly sharp too early.
        scores = torch.einsum("btsd,d->bts", normed, query) / math.sqrt(stacked.size(-1))
        weights = torch.softmax(scores, dim=2)
        return torch.einsum("bts,btsd->btd", weights, stacked)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def _residual_row_mix(self, current: torch.Tensor, row_q_proj: nn.Linear) -> torch.Tensor:
        current = self._rms_norm_tensor(current)
        q_row = self._split_heads(row_q_proj(current))
        kv_row = self._split_heads(current)
        seq_len = current.size(1)
        scores = torch.matmul(q_row, kv_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=current.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, kv_row)
        return self._merge_heads(context)

    def _attn_res_dual_axis_mix(
        self,
        embedding: torch.Tensor,
        history: List[torch.Tensor],
        depth_query: torch.Tensor,
        row_q_proj: nn.Linear,
    ) -> torch.Tensor:
        current = history[-1] if history else embedding
        return self._residual_row_mix(current, row_q_proj) + self._attn_res_mix(embedding, history, depth_query)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        if self.attention_type in {
            "attn_residuals",
            "attn_residuals_dual_axis",
            "dual_axis_full",
            "attn_residuals_value_reproj",
            "attn_residuals_value_reproj_normed",
            "attn_residuals_moe",
        }:
            embedding = x
            history: List[torch.Tensor] = []
            past_kv: List[KVCache] = []
            for idx, block in enumerate(self.blocks):
                if self.attention_type in {"attn_residuals_dual_axis", "dual_axis_full"}:
                    attn_input = self._attn_res_dual_axis_mix(
                        embedding,
                        history,
                        self.attn_res_queries[idx],
                        self.attn_res_row_q_projs[idx],
                    )
                else:
                    attn_input = self._attn_res_mix(embedding, history, self.attn_res_queries[idx])
                if isinstance(block.attn, DualAxisMemoryAttention):
                    attn_out, current_kv = block.attn(block.attn_norm(attn_input), past_kv=past_kv, past_states=history)
                else:
                    attn_out, current_kv = block.attn(block.attn_norm(attn_input), past_kv=past_kv)
                history.append(attn_out)
                past_kv.append(current_kv)

                if self.attention_type in {"attn_residuals_dual_axis", "dual_axis_full"}:
                    mlp_input = self._attn_res_dual_axis_mix(
                        embedding,
                        history,
                        self.mlp_res_queries[idx],
                        self.mlp_res_row_q_projs[idx],
                    )
                else:
                    mlp_input = self._attn_res_mix(embedding, history, self.mlp_res_queries[idx])
                mlp_out = block.mlp(block.mlp_norm(mlp_input))
                history.append(mlp_out)

            if self.attention_type in {"attn_residuals_dual_axis", "dual_axis_full"}:
                x = self._attn_res_dual_axis_mix(
                    embedding,
                    history,
                    self.final_res_query,
                    self.final_res_row_q_proj,
                )
            else:
                x = self._attn_res_mix(embedding, history, self.final_res_query)
            x = self.final_norm(x)
            return self.lm_head(x)

        past_kv: List[KVCache] = []
        past_ffn: List[FFNCache] = []
        past_states: List[torch.Tensor] = []
        for block in self.blocks:
            x, current_kv, current_ffn = block(
                x,
                past_kv=past_kv,
                past_ffn=past_ffn if past_ffn else None,
                past_states=past_states if past_states else None,
            )
            past_kv.append(current_kv)
            past_states.append(x)
            if current_ffn is not None:
                past_ffn.append(current_ffn)
        x = self.final_norm(x)
        return self.lm_head(x)
