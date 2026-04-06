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


class LightAttention(MultiHeadAttentionBase):
    """
    轻量注意力：仅保留 D 维掩码/缩放向量，剥离重排逻辑。
    参数量极低（仅 3D 每层），完全避免 O(D^2) 的计算和显存开销。
    属于最极端的消融验证对照组。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        # 仅用 3 个 D 维向量进行 QKV 特征的独立缩放
        self.w_qkv = nn.Parameter(torch.ones(3, d_model))

    def forward(self, x: torch.Tensor, past_kv=None):
        # 解包权重
        wq, wk, wv = self.w_qkv.unbind(dim=0)

        # 仅应用逐元素缩放，没有重排带来的耗时和显存负担
        q = x * wq
        k = x * wk
        v = x * wv

        # 变回多头张量
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 标准的 Attention 计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        
        return self.out_proj(self.merge_heads(out)), (k, v)


class SharedKVBaselineAttention(MultiHeadAttentionBase):
    """
    共享 KV 的 Baseline：
    - 每层独立 q_proj
    - 同层共享 k_proj / v_proj
    - 不使用跨层 memory（纯对照组，用于展示共享 KV 的负面效果）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        #实现有问题，不同层是不同改的共享wq，共享wk，但是q不等于k
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
        # Token 级别：合并投影以加速 (3倍效率)
        self.qkv_row_proj = nn.Linear(d_model, 3 * d_model)
        
        # Memory 级别：K/V 提取逻辑全局共享
        self.shared_memory_kv = shared_kv
        

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # 1. 提速：一次矩阵乘法解决 q_row, k_row, v_row
        q_row, k_row, v_row = self.qkv_row_proj(x).chunk(3, dim=-1)
        q_row = self.split_heads(q_row)
        k_row = self.split_heads(k_row)
        v_row = self.split_heads(v_row)

        q_col = q_row
        k_col = self.split_heads(self.shared_memory_kv.k_proj(x))
        v_col = self.split_heads(self.shared_memory_kv.v_proj(x))

        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(
            self.causal_mask(seq_len, x.device), float("-inf")
        )

        if past_kv is not None:
            past_keys, past_values = past_kv

            # 2. 提速：使用 einsum 避免 5D 张量广播带来的巨大显存和时间开销
            memory_scores = torch.einsum('b h s d, b h s l d -> b h s l', q_col, past_keys) / math.sqrt(self.head_dim)
            
            scores  = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))

            token_w = weights[..., :seq_len]
            mem_w   = weights[..., seq_len:]

            token_ctx = torch.matmul(token_w, v_row)
            # 3. 提速：使用 einsum 直接高效加权求和，避免 unsqueeze 广播
            mem_ctx   = torch.einsum('b h s l, b h s l d -> b h s d', mem_w, past_values)
            
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


class SharedCrossQProj(nn.Module):
    """
    全模型共享的多头 Q 投影（attn_residual_2d 专用）。
    Q 由当前位置输入投影得到（input-dependent），增强表达力；
    所有层共享同一个矩阵，避免参数量随层数线性增长。
    额外参数：D×D（共享一次）+ D（LayerNorm 各层各一份）。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D] → [B, H, S, head_dim]"""
        B, S, _ = x.shape
        return (
            self.q_proj(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )


class AttnResModule2D(nn.Module):
    """
    十字形多头注意力残差（改进版）：

      Q: 由当前层输入投影（input-dependent），所有层共享同一个 SharedCrossQProj
      K: LayerNorm(历史层输入)，无额外投影
      V: 历史层输入，无额外投影

      感受野（十字形）：
           s1   s2   s3    t
      L-1 │ ←   ←   ←   [x]  ← 顶行：当前层所有位置 s≤t（横向，S 个 Key）
      L-2 │              [↑]
      L-3 │              [↑]  ← 右列：所有层在位置 t（纵向）
       …  │              [↑]     每层存 x_mid + x 共 2 个状态 → 2L 个 Key

      Key 总数：2L + S
        - 纵向右列：2L 个（每层贡献 x_mid 和 x 两个亚层状态）
        - 横向顶行：S 个（当前层所有 s≤t 位置，带因果掩码）
        - 对比标准 AttnRes（纵向）：L 个
    """
    def __init__(self, d_model: int, shared_q: SharedCrossQProj):
        super().__init__()
        self.shared_q = shared_q
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, layer_history: list, current: torch.Tensor) -> torch.Tensor:
        if not layer_history:
            return current

        B, S, D = current.shape
        H, d    = self.shared_q.num_heads, self.shared_q.head_dim
        scale   = self.shared_q.scale

        # Q 由当前输入投影：[B, H, S_out, d]
        Q = self.shared_q.project(current)

        # ── 纵向（右列）：所有层在位置 t ──────────────────────────────
        V_col   = torch.stack(layer_history + [current], dim=0)    # [L+1, B, S, D]
        K_col   = self.norm(V_col)                                  # [L+1, B, S, D]
        L_p1    = V_col.size(0)
        # 分头：[L+1, B, H, S, d]
        K_col_h = K_col.view(L_p1, B, S, H, d).permute(0, 1, 3, 2, 4)
        V_col_h = V_col.view(L_p1, B, S, H, d).permute(0, 1, 3, 2, 4)
        # Q[b,h,t,d] · K_col[l,b,h,t,d] → [B, H, S, L+1]
        scores_col = torch.einsum(
            'b h t d, l b h t d -> b h t l', Q, K_col_h
        ) / scale

        # ── 横向（顶行）：当前层所有位置 s≤t ─────────────────────────
        K_row_h = self.norm(current).view(B, S, H, d).transpose(1, 2)  # [B, H, S_in, d]
        V_row_h = current.view(B, S, H, d).transpose(1, 2)             # [B, H, S_in, d]
        # Q[b,h,t,d] · K_row[b,h,s,d] → [B, H, S_out, S_in]
        scores_row = torch.einsum(
            'b h t d, b h s d -> b h t s', Q, K_row_h
        ) / scale
        # 因果掩码
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=current.device), diagonal=1
        )
        scores_row = scores_row.masked_fill(causal, float('-inf'))

        # ── 拼接 + 统一 Softmax over [L+1 col + S row] ──────────────
        scores_all  = torch.cat([scores_col, scores_row], dim=-1)  # [B, H, S, L+1+S]
        weights_all = scores_all.softmax(dim=-1)
        w_col = weights_all[..., :L_p1]   # [B, H, S, L+1]
        w_row = weights_all[..., L_p1:]   # [B, H, S, S]

        # 纵向聚合：Σ_l w_col * V_col[l,b,h,t,:]
        h_col = torch.einsum('b h t l, l b h t d -> b h t d', w_col, V_col_h)  # [B, H, S, d]
        # 横向聚合：Σ_s w_row * V_row[b,h,s,:]
        h_row = torch.einsum('b h t s, b h s d -> b h t d', w_row, V_row_h)    # [B, H, S, d]

        # 合并多头：[B, S, D]
        return (h_col + h_row).transpose(1, 2).contiguous().view(B, S, D)


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
        shared_cross_q: SharedCrossQProj = None,   # attn_residual_2d 专用
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
        self.use_attn_residual_2d = shared_cross_q is not None

        if use_attn_residual:
            self.attn_res_attn = AttnResModule(d_model)
            self.attn_res_mlp  = AttnResModule(d_model)
        elif shared_cross_q is not None:
            # 所有层共享同一个 SharedCrossQProj，各自有独立的 LayerNorm
            self.attn_res_attn = AttnResModule2D(d_model, shared_cross_q)
            self.attn_res_mlp  = AttnResModule2D(d_model, shared_cross_q)

    def forward(self, x: torch.Tensor, past_kv=None, layer_history=None):
        use_ar = (self.use_attn_residual or self.use_attn_residual_2d) and layer_history is not None

        if use_ar:
            # Bug 2 修复：layer_history[-1] 就是 x（上一个 block 的输出），
            # 不能再把 current=x 塞进去，否则历史里会重复计入。
            # 把 layer_history 拆成 history_before（不含 x）和 current（= x）。
            if layer_history:
                hist_for_attn, cur_for_attn = layer_history[:-1], layer_history[-1]
            else:
                hist_for_attn, cur_for_attn = [], x
            h = self.attn_res_attn(hist_for_attn, cur_for_attn)
            attn_out, kv_attn = self.attn(self.attn_norm(h), past_kv=past_kv)
            x = attn_out          # Attention 无标准残差（AttnRes 替代了跨层残差）
        else:
            attn_out, kv_attn = self.attn(self.attn_norm(x), past_kv=past_kv)
            x = x + attn_out

        # x_mid = Attention 输出 / FFN 输入
        x_mid = x

        # attn_res_mlp：x_mid 是新状态，不在 layer_history 里，直接传入不重复
        h = self.attn_res_mlp(layer_history, x_mid) if use_ar else x_mid

        # 亚层级记忆截点（SubLayer 模式）
        if self.extract_sublayers and self.shared_kv is not None:
            norm_x = self.mlp_norm(x_mid)
            k_ffn  = self.attn.split_heads(self.shared_kv.k_proj(norm_x))
            v_ffn  = self.attn.split_heads(self.shared_kv.v_proj(norm_x))
            current_kv = [kv_attn, (k_ffn, v_ffn)]
        else:
            current_kv = kv_attn

        mlp_out = self.mlp(self.mlp_norm(h))
        if self.use_attn_residual or self.use_attn_residual_2d:
            # Bug 1 修复：官方 AttnRes 在块边界时，MLP 输出加在 attn_out 上，
            # 不是完全无残差。对应 partial_block = partial_block + mlp_out
            # 其中 partial_block = attn_out = x_mid。
            x = mlp_out
        else:
            x = x + mlp_out

        if self.use_attn_residual_2d:
            return x, current_kv, x_mid
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
        # attn_residual_2d：全模型共享一个 Q 投影，所有层传入同一个对象
        shared_cross_q = (
            SharedCrossQProj(d_model, num_heads)
            if attention_type == "attn_residual_2d" else None
        )
        if shared_cross_q is not None:
            self.shared_cross_q = shared_cross_q  # 注册为子模块，参与参数追踪

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = self._build_block(
                attention_type, d_model, num_heads, mlp_ratio, dropout,
                shared_kv, shared_cross_q
            )
            self.blocks.append(block)

        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def _build_block(self, attention_type, d_model, num_heads, mlp_ratio, dropout,
                     shared_kv, shared_cross_q=None):
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
        elif attention_type == "light_attention":
            attn  = LightAttention(d_model, num_heads, dropout)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)

        elif attention_type == "attn_residual_2d":
            # 十字形多头注意力残差：Q 由输入投影，所有层共享同一个 SharedCrossQProj
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(
                d_model, num_heads, mlp_ratio, dropout, attn,
                shared_cross_q=shared_cross_q
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

        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        layer_history = []   # 供 attn_residual 使用

        for block in self.blocks:
            if self.attention_type in {"shared_kv_depth_memory", "shared_kv_depth_memory_dualq"}:
                x, current_kv = block(x, past_kv=past_kv)
                past_kv = self._append_past_kv(past_kv, current_kv)

            elif self.attention_type == "shared_kv_baseline":
                x, _ = block(x, past_kv=None)  # 不使用跨层记忆，直接传 None

            elif self.attention_type == "shared_kv_depth_memory_dualq_sublayer":
                x, current_kvs = block(x, past_kv=past_kv)
                for current_kv in current_kvs:
                    past_kv = self._append_past_kv(past_kv, current_kv)

            elif self.attention_type == "attn_residual":
                x, _ = block(x, past_kv=None, layer_history=layer_history)
                layer_history.append(x)

            elif self.attention_type == "attn_residual_2d":
                # 返回三元组：(x_final, kv, x_mid)
                # 将 x_mid（FFN 输入）和 x（块最终输出）都存入历史，深度伸展到 2L
                x, _, x_mid = block(x, past_kv=None, layer_history=layer_history)
                layer_history.append(x_mid)   # Attention 后、FFN 前的中间状态
                layer_history.append(x)        # FFN 后的最终状态

            else:
                x, _ = block(x, past_kv=None)

        x = self.final_norm(x)
        return self.lm_head(x)

    @staticmethod
    def _append_past_kv(
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        current_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = current_kv
        if past_kv is None:
            return k.unsqueeze(3), v.unsqueeze(3)
        past_keys, past_values = past_kv
        return (
            torch.cat([past_keys, k.unsqueeze(3)], dim=3),
            torch.cat([past_values, v.unsqueeze(3)], dim=3),
        )
    
