"""
ablation_models.py
==================
Layer-Depth-Attention ?

?
  - baseline                              :  Transformer ?
  - shared_kv_baseline                    :  KV ?Baseline?
  - shared_kv_depth_memory_dualq          : ?+ 
  - shared_kv_depth_memory_dualq_sublayer :  + FFN 
  - depth_memory_reuse_row_qkv            : ?q/k/v?
  - attn_residual                         : Kimi AttnRes?
  - light_attention                       : ?D 
  - light_attention_kfromrow               : ?K  D   
 Notebook ?
  import sys
  sys.path.insert(0, "/path/to/Layer-Depth-Attention/src")
  from layer_depth_attention.ablation_models import TinyDecoderLM
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# 
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
    """Shared K/V projector used across layers."""
    def __init__(self, d_model: int):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)


# ============================================================
# ?
# ============================================================

class BaselineAttention(MultiHeadAttentionBase):
    """Standard multi-head causal self-attention."""
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, x: torch.Tensor, past_kv=None, past_x=None):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self.out_proj(self.merge_heads(out)), (k, v)


class LightAttention(MultiHeadAttentionBase):
    """
    ?D ??
    ?3D  O(D^2) ?
    ?
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        #  3 ?D ?QKV ?
        self.w_qkv = nn.Parameter(torch.ones(3, d_model))
        

    def forward(self, x: torch.Tensor, past_kv=None,past_x = None):
        # 
        wq, wk, wv = self.w_qkv.unbind(dim=0)

        # ?
        if past_x is not None:
            q = past_x * wq
        else:
            q = x * wq
        k = x * wk
        v = x * wv

        # 
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # ?Attention 
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        
        return self.out_proj(self.merge_heads(out)), (k, v)


# class AgregateQ(MultiHeadAttentionBase):




class SharedKVBaselineAttention(MultiHeadAttentionBase):
    """
     KV ?Baseline?
    -  q_proj
    -  k_proj / v_proj
    - ?memory KV 
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        #wqwkqk
        self.shared_kv = shared_kv

    def forward(self, x: torch.Tensor, past_kv=None, past_x=None):
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
    ?Memory ?
    - ?(Row/Token) q_row, k_row, v_row ?
    - ?(Col/Depth) q_col ?k_proj/v_proj ?
    - ?Softmax ?
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, shared_kv: SharedKVProjector):
        super().__init__(d_model, num_heads, dropout)
        # Token ?(3?
        self.qkv_row_proj = nn.Linear(d_model, 3 * d_model)
        
        # Memory K/V 
        self.shared_memory_kv = shared_kv
        

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_x: Optional[torch.Tensor] = None,
    ):
        # 1. ?q_row, k_row, v_row
        q_row, k_row, v_row = self.qkv_row_proj(x).chunk(3, dim=-1)
        q_row = self.split_heads(q_row)
        k_row = self.split_heads(k_row)
        v_row = self.split_heads(v_row)

        q_col = q_row
        k_col = self.split_heads(self.shared_memory_kv.k_proj(x))
        v_col = self.split_heads(self.shared_memory_kv.v_proj(x))
        #  x  k_col ?v_col
        # k_col = self.split_heads(x)
        # v_col = self.split_heads(x)

        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(
            self.causal_mask(seq_len, x.device), float("-inf")
        )

        if past_kv is not None:
            past_keys, past_values = past_kv

            # 2.  einsum  5D 
            memory_scores = torch.einsum('b h s d, b h s l d -> b h s l', q_col, past_keys) / math.sqrt(self.head_dim)
            
            scores  = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))

            token_w = weights[..., :seq_len]
            mem_w   = weights[..., seq_len:]

            token_ctx = torch.matmul(token_w, v_row)
            # 3.  einsum ?unsqueeze 
            mem_ctx   = torch.einsum('b h s l, b h s l d -> b h s d', mem_w, past_values)
            
            out = token_ctx + mem_ctx
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            out = torch.matmul(weights, v_row)

        #  k_col / v_col ?
        return self.out_proj(self.merge_heads(out)), (k_col, v_col)


class DepthMemoryReuseRowQKVAttention(MultiHeadAttentionBase):
    """
     depth-memory ?
    - ?(Row/Token)  fused QKV ?
    - ?(Col/Depth)  row q ?
       row k/v?
    -  row k/v ?
      ?memory k_proj / v_proj?
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        self.qkv_row_proj = nn.Linear(d_model, 3 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_x: Optional[torch.Tensor] = None,
    ):
        q_row, k_row, v_row = self.qkv_row_proj(x).chunk(3, dim=-1)
        q_row = self.split_heads(q_row)
        k_row = self.split_heads(k_row)
        v_row = self.split_heads(v_row)

        # ?past_kv ?
        #  row k/v?k/v?
        q_col = q_row

        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(
            self.causal_mask(seq_len, x.device), float("-inf")
        )

        if past_kv is not None:
            past_keys, past_values = past_kv
            memory_scores = torch.einsum(
                "b h s d, b h s l d -> b h s l", q_col, past_keys
            ) / math.sqrt(self.head_dim)

            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))

            token_w = weights[..., :seq_len]
            mem_w = weights[..., seq_len:]

            token_ctx = torch.matmul(token_w, v_row)
            mem_ctx = torch.einsum("b h s l, b h s l d -> b h s d", mem_w, past_values)
            out = token_ctx + mem_ctx
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            out = torch.matmul(weights, v_row)

        #  row k/v ?
        return self.out_proj(self.merge_heads(out)), (k_row, v_row)


class DepthMemoryReuseHiddenStatesAttention(MultiHeadAttentionBase):
    """
    Same-position depth memory that reads cached hidden states instead of
    separate memory K/V projections. The row branch is standard fused QKV;
    the depth branch reuses q_row to query previously cached hidden states.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        self.qkv_row_proj = nn.Linear(d_model, 3 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_x: Optional[torch.Tensor] = None,
    ):
        q_row, k_row, v_row = self.qkv_row_proj(x).chunk(3, dim=-1)
        q_row = self.split_heads(q_row)
        k_row = self.split_heads(k_row)
        v_row = self.split_heads(v_row)

        q_col = q_row

        seq_len = x.size(1)
        token_scores = torch.matmul(q_row, k_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_scores = token_scores.masked_fill(
            self.causal_mask(seq_len, x.device), float("-inf")
        )

        if past_kv is not None:
            past_keys, past_values = past_kv
            memory_scores = torch.einsum(
                "b h s d, b h s l d -> b h s l", q_col, past_keys
            ) / math.sqrt(self.head_dim)

            scores = torch.cat([token_scores, memory_scores], dim=-1)
            weights = self.dropout(torch.softmax(scores, dim=-1))

            token_w = weights[..., :seq_len]
            mem_w = weights[..., seq_len:]

            token_ctx = torch.matmul(token_w, v_row)
            mem_ctx = torch.einsum("b h s l, b h s l d -> b h s d", mem_w, past_values)
            out = token_ctx + mem_ctx
        else:
            weights = self.dropout(torch.softmax(token_scores, dim=-1))
            out = torch.matmul(weights, v_row)

        return self.out_proj(self.merge_heads(out)), (k_row, v_row)






# ============================================================
# Kimi AttnRes
# ============================================================

class AttnResModule(nn.Module):
    """
    Kimi Attention ResidualsAttnRes
    - pseudo-query?
    -  LayerNorm?
    - ?Softmax
    https://github.com/MoonshotAI/Attention-Residuals
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

        # 1.  input-dependent ?Query
        # Q = self.pseudo_query.weight.squeeze(0) * current    # [B, S, D]
        Q = current

        # 2. ?Query ?K  sqrt(D) 
        logits = torch.einsum(
            'b s d, l b s d -> l b s',
            Q, K
        ) / math.sqrt(current.size(-1))                      # [L+1, B, S]
        weights = logits.softmax(dim=0)                      # softmax ?
        return torch.einsum('l b s, l b s d -> b s d', weights, V)


class SharedCrossQProj(nn.Module):
    """
     Q attn_residual_2d ?
    Q input-dependent
    ?
    DD+ DLayerNorm ?
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D] ?[B, H, S, head_dim]"""
        B, S, _ = x.shape
        return (
            self.q_proj(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )


class AttnResModule2D(nn.Module):
    """
    

      Q: input-dependent?SharedCrossQProj
      K: LayerNorm(?
      V: ?

      ?
           s1   s2   s3    t
      L-1 ??  ?  ?  [x]  ??stS ?Key?
      L-2 ?             []
      L-3 ?             []  ??t
       ? ?             []     ?x_mid + x ?2 ??2L ?Key

      Key ?L + S
        - ?L  x_mid ?x 
        - S ?st ?
        -  AttnResL ?
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

        # Q [B, H, S_out, d]
        Q = self.shared_q.project(current)

        #  ?t 
        V_col   = torch.stack(layer_history + [current], dim=0)    # [L+1, B, S, D]
        K_col   = self.norm(V_col)                                  # [L+1, B, S, D]
        L_p1    = V_col.size(0)
        # [L+1, B, H, S, d]
        K_col_h = K_col.view(L_p1, B, S, H, d).permute(0, 1, 3, 2, 4)
        V_col_h = V_col.view(L_p1, B, S, H, d).permute(0, 1, 3, 2, 4)
        # Q[b,h,t,d]  K_col[l,b,h,t,d] ?[B, H, S, L+1]
        scores_col = torch.einsum(
            'b h t d, l b h t d -> b h t l', Q, K_col_h
        ) / scale

        #  ?st 
        K_row_h = self.norm(current).view(B, S, H, d).transpose(1, 2)  # [B, H, S_in, d]
        V_row_h = current.view(B, S, H, d).transpose(1, 2)             # [B, H, S_in, d]
        # Q[b,h,t,d]  K_row[b,h,s,d] ?[B, H, S_out, S_in]
        scores_row = torch.einsum(
            'b h t d, b h s d -> b h t s', Q, K_row_h
        ) / scale
        # 
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=current.device), diagonal=1
        )
        scores_row = scores_row.masked_fill(causal, float('-inf'))

        #   +  Softmax over [L+1 col + S row] 
        scores_all  = torch.cat([scores_col, scores_row], dim=-1)  # [B, H, S, L+1+S]
        weights_all = scores_all.softmax(dim=-1)
        w_col = weights_all[..., :L_p1]   # [B, H, S, L+1]
        w_row = weights_all[..., L_p1:]   # [B, H, S, S]

        # _l w_col * V_col[l,b,h,t,:]
        h_col = torch.einsum('b h t l, l b h t d -> b h t d', w_col, V_col_h)  # [B, H, S, d]
        # _s w_row * V_row[b,h,s,:]
        h_row = torch.einsum('b h t s, b h s d -> b h t d', w_row, V_row_h)    # [B, H, S, d]

        # [B, S, D]
        return (h_col + h_row).transpose(1, 2).contiguous().view(B, S, D)


# ============================================================
# TransformerBlock
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
        shared_cross_q: SharedCrossQProj = None,   # attn_residual_2d 
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
            # ?SharedCrossQProj?LayerNorm
            self.attn_res_attn = AttnResModule2D(d_model, shared_cross_q)
            self.attn_res_mlp  = AttnResModule2D(d_model, shared_cross_q)

    def forward(self, x: torch.Tensor, past_kv=None, layer_history=None, past_x=None):
        use_ar = (self.use_attn_residual or self.use_attn_residual_2d) and layer_history is not None
        
        current_input_x = x.clone()

        if use_ar:
            # Bug 2 layer_history[-1]  x?block ?
            #  current=x ?
            # ?layer_history  history_before?x current? x?
            if layer_history:
                hist_for_attn, cur_for_attn = layer_history[:-1], layer_history[-1]
            else:
                hist_for_attn, cur_for_attn = [], x
            h = self.attn_res_attn(hist_for_attn, cur_for_attn)
            attn_out, kv_attn = self.attn(self.attn_norm(h), past_kv=past_kv, past_x=past_x)
            x = attn_out          # Attention AttnRes 
        else:
            attn_out, kv_attn = self.attn(self.attn_norm(x), past_kv=past_kv, past_x=past_x)
            x = x + attn_out

        # x_mid = Attention  / FFN 
        x_mid = x

        # attn_res_mlpx_mid  layer_history ?
        h = self.attn_res_mlp(layer_history, x_mid) if use_ar else x_mid

        # SubLayer ?
        if self.extract_sublayers and self.shared_kv is not None:
            norm_x = self.mlp_norm(x_mid)
            k_ffn  = self.attn.split_heads(self.shared_kv.k_proj(norm_x))
            v_ffn  = self.attn.split_heads(self.shared_kv.v_proj(norm_x))
            current_kv = [kv_attn, (k_ffn, v_ffn)]
        elif self.extract_sublayers and self.shared_kv is None:
            x_mid_heads = self.attn.split_heads(x_mid)
            current_kv = [(x_mid_heads, x_mid_heads)]
        else:
            current_kv = kv_attn

        mlp_out = self.mlp(self.mlp_norm(h))
        if self.use_attn_residual or self.use_attn_residual_2d:
            # Bug 1 ?AttnRes MLP  attn_out 
            # ?partial_block = partial_block + mlp_out
            #  partial_block = attn_out = x_mid?
            x = mlp_out
        else:
            x = x + mlp_out

        if self.extract_sublayers and self.shared_kv is None:
            x_heads = self.attn.split_heads(x)
            current_kv.append((x_heads, x_heads))

        if self.use_attn_residual_2d:
            return x, current_kv, x_mid
        
        return x, current_kv, current_input_x



# ============================================================
# TinyDecoderLM?
# ============================================================

class TinyDecoderLM(nn.Module):
    """
     Decoder-only ?

    attention_type 
      "baseline"                              -  Transformer
      "shared_kv_baseline"                    -  KV 
      "shared_kv_depth_memory_dualq"          - ?
      "shared_kv_depth_memory_dualq_sublayer" -  + ?
      "depth_memory_reuse_row_qkv"            - ?row q/k/v?
      "depth_memory_hidden_states_sublayer"   - cache residual-merged sublayer outputs
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
        # attn_residual_2d?Q ?
        shared_cross_q = (
            SharedCrossQProj(d_model, num_heads)
            if attention_type == "attn_residual_2d" else None
        )
        if shared_cross_q is not None:
            self.shared_cross_q = shared_cross_q  # ?

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

        elif attention_type == "depth_memory_reuse_row_qkv":
            attn  = DepthMemoryReuseRowQKVAttention(d_model, num_heads, dropout)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)

        elif attention_type == "depth_memory_hidden_states_sublayer":
            attn  = DepthMemoryReuseHiddenStatesAttention(d_model, num_heads, dropout)
            block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, attn)
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
            # Q ?SharedCrossQProj
            attn  = BaselineAttention(d_model, num_heads, dropout)
            block = TransformerBlock(
                d_model, num_heads, mlp_ratio, dropout, attn,
                shared_cross_q=shared_cross_q
            )
        else:
            raise ValueError(f"?attention_type: {attention_type!r}")

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
        layer_history = []   # ?attn_residual 


        past_x = None        #  None

        for block in self.blocks:
            if self.attention_type in {
                "shared_kv_depth_memory",
                "shared_kv_depth_memory_dualq",
                "depth_memory_reuse_row_qkv",
            }:
                x, current_kv, past_x = block(x, past_kv=past_kv, past_x=past_x)
                past_kv = self._append_past_kv(past_kv, current_kv)

            elif self.attention_type == "shared_kv_baseline":
                x, _, past_x = block(x, past_kv=None, past_x=past_x)

            elif self.attention_type in {"shared_kv_depth_memory_dualq_sublayer", "depth_memory_hidden_states_sublayer"}:
                x, current_kvs, past_x = block(x, past_kv=past_kv, past_x=past_x)
                for current_kv in current_kvs:
                    past_kv = self._append_past_kv(past_kv, current_kv)

            elif self.attention_type == "attn_residual":
                x, _, past_x = block(x, past_kv=None, layer_history=layer_history, past_x=past_x)
                layer_history.append(x)

            elif self.attention_type == "attn_residual_2d":
                x, _, x_mid = block(x, past_kv=None, layer_history=layer_history, past_x=past_x)
                layer_history.append(x_mid)
                layer_history.append(x)
                past_x = x

            else:
                x, _, past_x = block(x, past_kv=None, past_x=past_x)

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
    
