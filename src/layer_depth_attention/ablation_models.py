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
# Kimi AttnRes：注意力残差替代标准残差连接（正确实现）
# ============================================================

class AttnResModule(nn.Module):
    """
    Kimi Attention Residuals（AttnRes）官方正确实现：
    - 使用固定的可学习伪向量（pseudo-query），与当前输入无关
    - 对历史层输出先做 LayerNorm，防止深层量级支配
    - 在层维度做 Softmax，加权聚合所有历史层输出
    参考：https://github.com/MoonshotAI/Attention-Residuals
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.pseudo_query = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, layer_history: list, current: torch.Tensor) -> torch.Tensor:
        if not layer_history:
            return current
        V = torch.stack(layer_history + [current], dim=0)   # [L+1, B, S, D]
        K = self.norm(V)
        logits = torch.einsum(
            'd, l b s d -> l b s',
            self.pseudo_query.weight.squeeze(0), K
        )                                                    # [L+1, B, S]
        weights = logits.softmax(dim=0)                      # softmax 层维度
        return torch.einsum('l b s, l b s d -> b s d', weights, V)


class AttnResModule2D(nn.Module):
    """
    十字形（Cross）注意力残差：
      - 纵向（右列）：所有层在当前 Token t 位置的表示（L+1 个 Key）← 等价于标准 AttnRes
      - 横向（顶行）：最新一层在所有 s≤t 位置的表示（S 个 Key）←  新增横向感受野

    十字形 Key 总数：(L+1) + S - 1 = L+S 个（远少于全网格 L×S）
    参数量与 AttnRes 完全相同：一个伪向量 + LayerNorm，不做额外投影。

         s1   s2   s3    t
    L-1 │ ←   ←   ←   [x]   ← 顶行：最新层所有历史 Token
    L-2 │              [↑]
    L-3 │              [↑]   ← 右列：所有层当前 Token t
     …  │              [↑]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.pseudo_query = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, layer_history: list, current: torch.Tensor) -> torch.Tensor:
        if not layer_history:
            return current

        B, S, D = current.shape
        w = self.pseudo_query.weight.squeeze(0)   # [D]

        # ── 纵向（右列）：所有层在位置 t 的表示 ──────────────────────
        # V_col[l, b, t, :] = layer_history[l][b, t, :]
        V_col = torch.stack(layer_history + [current], dim=0)  # [L+1, B, S, D]
        K_col = self.norm(V_col)                                # [L+1, B, S, D]
        # 只取每个位置 t 对应的列：einsum 沿层维度打分
        scores_col = torch.einsum('d, l b s d -> b s l', w, K_col)  # [B, S, L+1]

        # ── 横向（顶行）：最新层在所有 s≤t 位置的表示 ──────────────
        # 最新层 = current（第 L 层输入）
        K_row = self.norm(current)                              # [B, S_in, D]
        scores_row = torch.einsum('d, b s d -> b s', w, K_row)  # [B, S_in]
        # 扩展为每个输出 Token t 的视角：[B, S_out, S_in]
        scores_row = scores_row.unsqueeze(1).expand(B, S, S)    # [B, S_out, S_in]
        # 因果掩码：s_in > t 的位置不可见
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=current.device), diagonal=1
        )
        scores_row = scores_row.masked_fill(causal, float('-inf'))

        # ── 拼接并统一 Softmax ─────────────────────────────────────
        # col: [B, S_out, L+1]，row: [B, S_out, S_in]
        # 拼接成 [B, S_out, L+1+S]，在最后一维统一 Softmax
        scores_all  = torch.cat([scores_col, scores_row], dim=-1)  # [B, S, L+1+S]
        weights_all = scores_all.softmax(dim=-1)                   # [B, S, L+1+S]

        w_col = weights_all[..., :V_col.size(0)]   # [B, S, L+1]
        w_row = weights_all[..., V_col.size(0):]   # [B, S, S]

        # 纵向聚合：Σ_l w_col[b,t,l] * V_col[l,b,t,:]
        h_col = torch.einsum('b s l, l b s d -> b s d', w_col, V_col)   # [B, S, D]

        # 横向聚合：Σ_{s_in≤t} w_row[b,t,s_in] * current[b,s_in,:]
        h_row = torch.einsum('b t s, b s d -> b t d', w_row, current)   # [B, S, D]

        return h_col + h_row


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
        use_attn_residual_2d: bool = False,
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
        self.shared_kv         = shared_kv
        self.extract_sublayers = False
        self.use_attn_residual    = use_attn_residual
        self.use_attn_residual_2d = use_attn_residual_2d

        if use_attn_residual:
            self.attn_res_attn = AttnResModule(d_model)
            self.attn_res_mlp  = AttnResModule(d_model)
        elif use_attn_residual_2d:
            self.attn_res_attn = AttnResModule2D(d_model)
            self.attn_res_mlp  = AttnResModule2D(d_model)

    def forward(self, x: torch.Tensor, past_kv=None, layer_history=None):
        use_ar = (self.use_attn_residual or self.use_attn_residual_2d) and layer_history is not None

        if use_ar:
            h = self.attn_res_attn(layer_history, x)
            attn_out, kv_attn = self.attn(self.attn_norm(h), past_kv=past_kv)
            x = attn_out   # 无残差
        else:
            attn_out, kv_attn = self.attn(self.attn_norm(x), past_kv=past_kv)
            x = x + attn_out

        h = self.attn_res_mlp(layer_history, x) if use_ar else x

        # 亚层级记忆截点（SubLayer 模式）
        if self.extract_sublayers and self.shared_kv is not None:
            norm_x = self.mlp_norm(x)
            k_ffn  = self.attn.split_heads(self.shared_kv.k_proj(norm_x))
            v_ffn  = self.attn.split_heads(self.shared_kv.v_proj(norm_x))
            current_kv = [kv_attn, (k_ffn, v_ffn)]
        else:
            current_kv = kv_attn

        mlp_out = self.mlp(self.mlp_norm(h))
        if self.use_attn_residual:
            # ✅ AttnRes 模式：不用标准残差，AttnRes 已经聚合了历史（含当前 x）
            x = mlp_out
        else:
            x = x + mlp_out
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
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(
                d_model, num_heads, mlp_ratio, dropout, attn,
                use_attn_residual=True
            )

        elif attention_type == "attn_residual_2d":
            # 横纵扙2D AttnRes：所有层的所有 Token 作为记忆库
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(
                d_model, num_heads, mlp_ratio, dropout, attn,
                use_attn_residual_2d=True
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

            elif self.attention_type in {"attn_residual", "attn_residual_2d"}:
                x, _ = block(x, past_kv=None, layer_history=layer_history)
                layer_history.append(x)

            else:
                x, _ = block(x, past_kv=None)

        x = self.final_norm(x)
        return self.lm_head(x)
