import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


KVCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class VisionAttentionBase(nn.Module):
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
        split = self.num_heads * self.head_dim
        k_weight = self.qkv_proj.weight[split : 2 * split]
        v_weight = self.qkv_proj.weight[2 * split :]
        k_bias = self.qkv_proj.bias[split : 2 * split]
        v_bias = self.qkv_proj.bias[2 * split :]
        k = F.linear(x, k_weight, k_bias)
        v = F.linear(x, v_weight, v_bias)
        return self._split_projected(k), self._split_projected(v)


class VisionSelfAttention(VisionAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        del past_kv
        q, k, v = self._split_qkv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class VisionValueReprojNormedAttention(VisionAttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        q, k, v = self._split_qkv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if past_kv:
            seq_len = x.size(1)
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
            all_scores = torch.cat([scores, memory_scores], dim=-1)
            weights = torch.softmax(all_scores, dim=-1)
            weights = self.dropout(weights)
            token_weights = weights[..., : x.size(1)]
            memory_weights = weights[..., x.size(1) :]
            token_context = torch.matmul(token_weights, v)
            memory_context = (memory_weights.unsqueeze(-1) * reproj_values).sum(dim=3)
            attn = token_context + memory_context
        else:
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        return self.out_proj(self._merge_heads(attn)), (k, v, q)


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        attention_type: str,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        if attention_type == "baseline":
            self.attn = VisionSelfAttention(d_model, num_heads, dropout)
        elif attention_type == "depth_memory_value_reproj_normed":
            self.attn = VisionValueReprojNormedAttention(d_model, num_heads, dropout)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, current_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, current_kv


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, d_model: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 100,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_type: str = "baseline",
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, d_model)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(x + self.pos_emb[:, : x.size(1)])

        past_kv: List[KVCache] = []
        for block in self.blocks:
            x, current_kv = block(x, past_kv if past_kv else None)
            past_kv.append(current_kv)

        x = self.norm(x)
        return self.head(x[:, 0])
