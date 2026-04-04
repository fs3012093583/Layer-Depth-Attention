"""
ablation_models.py
==================
Layer-Depth-Attention 消融实验模型库

包含以下模型变体：
  - baseline                              : 标准 Transformer 注意力
  - shared_kv_baseline                    : 共享 KV 的 Baseline（无跨层记忆）
  - shared_kv_depth_memory_dualq          : 双轴注意力（行独立 + 列共享记忆）
  - shared_kv_depth_memory_dualq_sublayer : 双轴 + 亚层级记忆（FFN 前截点分离）
  - attn_residual                         : Kimi AttnRes（注意力替代残差连接）

用法（在 Notebook 中引用）：
  import sys
  sys.path.insert(0, "/path/to/Layer-Depth-Attention/src")
  from layer_depth_attention.ablation_models import TinyDecoderLM
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# 基础组件
# ============================================================

class MultiHeadAttentionBase(nn.Module):
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
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )


class SharedKVProjector(nn.Module):
    """全局唯一的 K/V 投影器，被所有层共享。"""
    def __init__(self, d_model: int):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)


# ============================================================
# 注意力变体
# ============================================================

class BaselineAttention(MultiHeadAttentionBase):
    """标准 Multi-Head Self-Attention（每层完全独立）。"""
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, x: torch.Tensor, past_kv=None):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self.out_proj(self.merge_heads(out)), (k, v)


class SharedKVBaselineAttention(MultiHeadAttentionBase):
    """
    共享 KV 的 Baseline：
    - 每层独立 q_proj
    - 跨层共享 k_proj / v_proj
    - 不使用跨层 memory（纯对照组，用于展示共享 KV 的负面效果）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.shared_kv = shared_kv

    def forward(self, x: torch.Tensor, past_kv=None):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.shared_kv.k_proj(x))
        v = self.split_heads(self.shared_kv.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self.out_proj(self.merge_heads(out)), (k, v)


class SharedKVDepthMemoryDualQAttention(MultiHeadAttentionBase):
    """
    核心创新：双轴混合 Memory 注意力
    - 行 (Row/Token) 注意力：q_row, k_row, v_row 完全层独立，保留原生深模型算力。
    - 列 (Col/Depth) 注意力：q_col 独立检索，历史特征用全局共享的 k_proj/v_proj 归档。
    - 横纵两路分数拼接后经过同一个 Softmax 竞争，实现统一资源分配。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        # Token 级别：全部层间独立，不共享
        self.q_row_proj = nn.Linear(d_model, d_model)
        self.k_row_proj = nn.Linear(d_model, d_model)
        self.v_row_proj = nn.Linear(d_model, d_model)
        # Memory 级别：Q 独立，K/V 提取逻辑全局共享
        self.q_col_proj = nn.Linear(d_model, d_model)
        self.shared_memory_kv = shared_kv

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        q_row = self.split_heads(self.q_row_proj(x))
        k_row = self.split_heads(self.k_row_proj(x))
        v_row = self.split_heads(self.v_row_proj(x))

        q_col = self.split_heads(self.q_col_proj(x))
        k_col = self.split_heads(self.shared_memory_kv.k_proj(x))
        v_col = self.split_heads(self.shared_memory_kv.v_proj(x))

        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(
            self.causal_mask(seq_len, x.device), float("-inf")
        )

        if past_kv:
            past_keys   = torch.stack([item[0] for item in past_kv], dim=3)
            past_values = torch.stack([item[1] for item in past_kv], dim=3)

            memory_scores = (q_col.unsqueeze(3) * past_keys).sum(dim=-1) / math.sqrt(self.head_dim)
            scores  = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))

            token_w = weights[..., :seq_len]
            mem_w   = weights[..., seq_len:]

            token_ctx = torch.matmul(token_w, v_row)
            mem_ctx   = (mem_w.unsqueeze(-1) * past_values).sum(dim=3)
            out = token_ctx + mem_ctx
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            out = torch.matmul(weights, v_row)

        # 只把 k_col / v_col 送入历史档案库
        return self.out_proj(self.merge_heads(out)), (k_col, v_col)


# ============================================================
# Kimi AttnRes：注意力残差替代标准残差连接
# ============================================================

class AttentionResidualBlock(nn.Module):
    """
    Kimi Attention Residuals（AttnRes）：
    用轻量注意力动态聚合所有历史层输出，替代传统固定权重的残差连接。
    每层通过学习到的权重决定应该从哪些历史层汲取信息。
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = d_model // num_heads
        self.q_proj     = nn.Linear(d_model, d_model)
        self.scale      = math.sqrt(self.head_dim)

    def forward(self, x_current: torch.Tensor, layer_history: list) -> torch.Tensor:
        """
        x_current    : 当前层处理后的特征 [B, S, D]
        layer_history: 所有历史层输出的 List，每个元素 [B, S, D]
        """
        if not layer_history:
            return x_current  # 第 0 层无历史，直接返回

        B, S, D = x_current.shape
        past     = torch.stack(layer_history, dim=2)          # [B, S, L, D]
        num_past = past.size(2)

        q = self.q_proj(x_current)                            # [B, S, D]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]

        k = past.view(B, S, num_past, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4)                          # [B, H, S, L, d]

        scores  = (q.unsqueeze(3) * k).sum(dim=-1) / self.scale  # [B, H, S, L]
        weights = torch.softmax(scores, dim=-1)

        out = (weights.unsqueeze(-1) * k).sum(dim=3)          # [B, H, S, d]
        out = out.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]
        return out


# ============================================================
# TransformerBlock（统一支持所有变体）
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        attention: nn.Module,
        shared_kv: SharedKVProjector = None,
        use_attn_residual: bool = False,
        no_attn_res_skip: bool = False,   # 是否取消 attention 的残差连接（实验用）
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn      = attention
        self.mlp_norm  = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.shared_kv        = shared_kv
        self.extract_sublayers = False
        self.no_attn_res_skip  = no_attn_res_skip

        # Kimi AttnRes 模式
        self.use_attn_residual = use_attn_residual
        if use_attn_residual:
            self.attn_res = AttentionResidualBlock(d_model, num_heads=min(4, num_heads))

    def forward(self, x: torch.Tensor, past_kv=None, layer_history=None):
        attn_out, kv_attn = self.attn(self.attn_norm(x), past_kv=past_kv)

        if self.use_attn_residual and layer_history is not None:
            # Kimi AttnRes：用注意力动态权重聚合所有历史层
            x = self.attn_res(x + attn_out, layer_history)
        elif self.no_attn_res_skip:
            # 实验：完全跳过 attention 残差连接
            x = attn_out
        else:
            # 标准残差
            x = x + attn_out

        # 亚层级记忆截点（FFN 前）
        if self.extract_sublayers and self.shared_kv is not None:
            norm_x = self.mlp_norm(x)
            k_ffn  = self.attn.split_heads(self.shared_kv.k_proj(norm_x))
            v_ffn  = self.attn.split_heads(self.shared_kv.v_proj(norm_x))
            current_kv = [kv_attn, (k_ffn, v_ffn)]
        else:
            current_kv = kv_attn

        x = x + self.mlp(self.mlp_norm(x))
        return x, current_kv


# ============================================================
# TinyDecoderLM（主模型，统一入口）
# ============================================================

class TinyDecoderLM(nn.Module):
    """
    小型 Decoder-only 语言模型，支持多种注意力变体。

    attention_type 可选值：
      "baseline"                              - 标准 Transformer
      "shared_kv_baseline"                    - 共享 KV 基线
      "shared_kv_depth_memory_dualq"          - 双轴注意力
      "shared_kv_depth_memory_dualq_sublayer" - 双轴 + 亚层级记忆
      "attn_residual"                         - Kimi AttnRes
    """

    SHARED_KV_TYPES = {
        "shared_kv_baseline",
        "shared_kv_depth_memory",
        "shared_kv_depth_memory_dualq",
        "shared_kv_depth_memory_dualq_sublayer",
    }

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
        self.use_pos_emb    = use_pos_emb

        self.token_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Embedding(max_seq_len, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout    = nn.Dropout(dropout)

        shared_kv = SharedKVProjector(d_model) if attention_type in self.SHARED_KV_TYPES else None

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = self._build_block(
                attention_type, d_model, num_heads, mlp_ratio, dropout, shared_kv
            )
            self.blocks.append(block)

        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def _build_block(self, attention_type, d_model, num_heads, mlp_ratio, dropout, shared_kv):
        if attention_type == "baseline":
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)

        elif attention_type == "shared_kv_baseline":
            attn  = SharedKVBaselineAttention(d_model, num_heads, dropout, shared_kv)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)

        elif attention_type == "shared_kv_depth_memory_dualq":
            attn  = SharedKVDepthMemoryDualQAttention(d_model, num_heads, dropout, shared_kv)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)

        elif attention_type == "shared_kv_depth_memory_dualq_sublayer":
            attn  = SharedKVDepthMemoryDualQAttention(d_model, num_heads, dropout, shared_kv)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn, shared_kv)
            block.extract_sublayers = True

        elif attention_type == "attn_residual":
            # Kimi AttnRes：Baseline 注意力 + 注意力残差替代标准残差
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(
                d_model, num_heads, mlp_ratio, dropout, attn,
                use_attn_residual=True
            )
        else:
            raise ValueError(f"未知的 attention_type: {attention_type!r}")

        return block

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
            x   = x + self.pos_emb(pos)
        x = self.dropout(x)

        past_kv       = []
        layer_history = []   # 供 attn_residual 使用

        for block in self.blocks:
            if self.attention_type in {"shared_kv_depth_memory", "shared_kv_depth_memory_dualq"}:
                x, current_kv = block(x, past_kv=past_kv)
                past_kv.append(current_kv)

            elif self.attention_type == "shared_kv_depth_memory_dualq_sublayer":
                x, current_kvs = block(x, past_kv=past_kv)
                past_kv.extend(current_kvs)   # 把 [kv_attn, kv_ffn] 全部展开

            elif self.attention_type == "attn_residual":
                x, _ = block(x, past_kv=None, layer_history=layer_history)
                layer_history.append(x.detach())  # 存入历史（detach 防止梯度爆链）

            else:
                x, _ = block(x, past_kv=None)

        x = self.final_norm(x)
        return self.lm_head(x)
