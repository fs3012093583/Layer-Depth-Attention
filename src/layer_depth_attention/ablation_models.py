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
    横纵扙3D注意力残差：把所有层的所有 Token 位置都列为记忆库。

    与标准 AttnRes 的区别：
      AttnRes  → 每个 Token 只回濙自身纵向历史（L 个 Key）
      AttnRes2D→ 每个 Token 可回濙历史所有层的所有早于它的 Token（L×S 个 Key）

    参数量与 AttnRes 相同：一个伪向量 + LayerNorm。
    V 和 K 直接用历史层输入，不做额外投影。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.pseudo_query = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, layer_history: list, current: torch.Tensor) -> torch.Tensor:
        if not layer_history:
            return current

        B, S, D = current.shape
        V = torch.stack(layer_history + [current], dim=0)  # [L+1, B, S_in, D]
        K = self.norm(V)                                    # [L+1, B, S_in, D]
        L_p1 = V.size(0)

        # 用伪向量对所有 (layer, position) 对打分
        w = self.pseudo_query.weight.squeeze(0)             # [D]
        scores_lbs = torch.einsum('d, l b s d -> l b s', w, K)  # [L+1, B, S_in]

        # 展开到每个输出 Token 的视角: [B, S_out, L+1, S_in]
        # scores 不依赖 S_out，不需要拷贝，用 expand 节省内存
        scores_2d = scores_lbs.permute(1, 2, 0)            # [B, S_in, L+1]
        scores_2d = scores_2d.unsqueeze(1).expand(B, S, S, L_p1)  # [B, S_out, S_in, L+1]
        scores_2d = scores_2d.permute(0, 1, 3, 2)          # [B, S_out, L+1, S_in]

        # 因果掩码: 输出 Token t 不能关注输入 s > t
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=current.device), diagonal=1
        )                                                   # [S_out, S_in]
        causal_4d = causal.unsqueeze(0).unsqueeze(2).expand(B, -1, L_p1, -1)
        scores_2d = scores_2d.masked_fill(causal_4d, float('-inf'))

        # 展平 (L+1, S_in) 成一维，统一做 Softmax
        scores_flat  = scores_2d.reshape(B, S, L_p1 * S)   # [B, S_out, (L+1)*S_in]
        weights_flat = scores_flat.softmax(dim=-1)          # [B, S_out, (L+1)*S_in]
        weights_2d   = weights_flat.reshape(B, S, L_p1, S) # [B, S_out, L+1, S_in]

        # 加权聚合: h[b,t,:] = Σ_{l,s<=t} w_{l,s} * V[l,b,s,:]
        V_perm = V.permute(1, 0, 2, 3)                      # [B, L+1, S_in, D]
        h = torch.einsum('b t l s, b l s d -> b t d', weights_2d, V_perm)
        return h


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
