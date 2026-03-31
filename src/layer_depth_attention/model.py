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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class LayerDepthMemoryAttention(MultiHeadAttentionBase):
    #‘’通过在注意力计算中直接拼接当前层的token scores和过去层的memory scores来实现层深度记忆机制，使用一个q
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        batch_size, _, seq_len, _ = q.shape

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:

            past_keys = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)

            # 使用相同的q来计算token_scores和memory_scores
            memory_scores = (q.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            
            #直接拼接token_scores和memory_scores，进行softmax
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
    #把过去的values进行线性变换后再计算memory_scores，其他部分同LayerDepthMemoryAttention，但是横纵使用不同的w
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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

            #同一个q
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


class LayerDepthDirectKVDualQAttention(MultiHeadAttentionBase):
    # 行内使用当前层 q_row，同列历史使用单独学习的 q_col。
    # 历史 k/v 直接复用，不再做任何重投影。
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

        # 当前层标准 causal attention，仍然由 q_row 查询当前层 k。
        token_scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            # 同列历史部分直接复用旧层已经算好的 k/v，只额外学习 q_col。
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


class LayerDepthDirectKVQMixAttention(MultiHeadAttentionBase):
    # 当前层仍保留 q_row/k/v。
    # 同列历史部分先得到 q_row 和 q_col，再用一个两路注意力把它们混成 q_mix。
    # 历史 k/v 直接复用，不做任何重投影。
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)
        self.column_q_proj = nn.Linear(d_model, d_model)

    def _mix_queries(self, q_row: torch.Tensor, q_col: torch.Tensor) -> torch.Tensor:
        # 用两路注意力决定 q_row / q_col 在 memory 查询里的占比。
        query_bank = torch.stack([q_row, q_col], dim=-2)
        mix_scores = torch.matmul(q_row.unsqueeze(-2), query_bank.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mix_weights = torch.softmax(mix_scores, dim=-1)
        mixed_query = torch.matmul(mix_weights, query_bank).squeeze(-2)
        return mixed_query

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q_row, k, v = self._split_qkv(x)
        q_col = self._split_projected(self.column_q_proj(x))
        q_mix = self._mix_queries(q_row, q_col)
        seq_len = x.size(1)

        # 当前层 token attention 仍使用标准的 q_row。
        token_scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        token_scores = token_scores.masked_fill(causal_mask, float("-inf"))

        if past_kv:
            # 同列历史直接复用旧层 k/v，memory 查询改用注意力混合后的 q_mix。
            past_keys = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)
            memory_scores = (q_mix.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
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


class LayerDepthValueReprojDualQAttention(MultiHeadAttentionBase):
    #双q，一个用于计算token_scores，一个用于计算memory_scores
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
    #加归一化
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
    #浅层kVQ都看作输入做翻倍线性变换，使用一个q
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
    #二维注意力机制，单q
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        seq_len = x.size(1)
        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
    # 在每个位置上使用一个路由器网络为输入分配权重，并选择一个专家进行计算，最后将专家输出乘以对应的权重作为最终输出
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
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
        elif attention_type == "depth_memory":
            self.attn = LayerDepthMemoryAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj":
            self.attn = LayerDepthValueReprojAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj_normed":
            self.attn = LayerDepthValueReprojNormedDualQAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_directkv_dualq":
            self.attn = LayerDepthDirectKVDualQAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_directkv_qmix":
            self.attn = LayerDepthDirectKVQMixAttention(d_model, num_heads, dropout)
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
    ) -> Tuple[torch.Tensor, KVCache, Optional[FFNCache]]:
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
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        residual_attention_types = {
            "attn_residuals",
            "attn_residuals_value_reproj",
            "attn_residuals_value_reproj_normed",
            "attn_residuals_moe",
        }
        if attention_type == "attn_residuals":
            block_attention_type = "baseline"
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
        scores = torch.einsum("btsd,d->bts", normed, query)
        weights = torch.softmax(scores, dim=2)
        return torch.einsum("bts,btsd->btd", weights, stacked)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        if self.attention_type in {
            "attn_residuals",
            "attn_residuals_value_reproj",
            "attn_residuals_value_reproj_normed",
            "attn_residuals_moe",
        }:
            embedding = x
            history: List[torch.Tensor] = []
            past_kv: List[KVCache] = []
            for idx, block in enumerate(self.blocks):
                attn_input = self._attn_res_mix(embedding, history, self.attn_res_queries[idx])
                attn_out, current_kv = block.attn(block.attn_norm(attn_input), past_kv=past_kv)
                history.append(attn_out)
                past_kv.append(current_kv)

                mlp_input = self._attn_res_mix(embedding, history, self.mlp_res_queries[idx])
                mlp_out = block.mlp(block.mlp_norm(mlp_input))
                history.append(mlp_out)

            x = self._attn_res_mix(embedding, history, self.final_res_query)
            x = self.final_norm(x)
            return self.lm_head(x)

        past_kv: List[KVCache] = []
        past_ffn: List[FFNCache] = []
        for block in self.blocks:
            x, current_kv, current_ffn = block(
                x,
                past_kv=past_kv,
                past_ffn=past_ffn if past_ffn else None,
            )
            past_kv.append(current_kv)
            if current_ffn is not None:
                past_ffn.append(current_ffn)
        x = self.final_norm(x)
        return self.lm_head(x)
