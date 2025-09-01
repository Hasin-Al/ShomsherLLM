import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from DeRoPE import DeRoPE


# Mulithead latent attention
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, rope_dim, compressed_dim, dropout, bias_qkv=False):
        super().__init__()
        assert rope_dim % 2 == 0, "rope_dim must be even"
        self.d_out     = d_out
        self.rope_dim  = rope_dim
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.split_dim = self.head_dim - self.rope_dim

        # KV compression
        self.W_dkv = nn.Linear(d_out, compressed_dim, bias=bias_qkv)
        self.W_dq  = nn.Linear(d_out, compressed_dim, bias=bias_qkv)

        # Expansion
        self.W_uk = nn.Linear(compressed_dim, num_heads * self.split_dim, bias=bias_qkv)
        self.W_uv = nn.Linear(compressed_dim, num_heads * self.head_dim,  bias=bias_qkv)
        self.W_uq = nn.Linear(compressed_dim, num_heads * self.split_dim, bias=bias_qkv)
        self.W_qr = nn.Linear(compressed_dim, num_heads * self.rope_dim,  bias=bias_qkv)
        self.W_kr = nn.Linear(compressed_dim, num_heads * self.rope_dim,  bias=bias_qkv)

        # RoPE
        self.rope   = DeRoPE(rope_dim)
        self.output = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_causal_mask=True,seq_start=0):
        """
        x: [B, L, d_out]
        past_kv: tuple(k_cached, v_cached) where
                 k_cached: [B, L_past, H, head_dim]
                 v_cached: [B, L_past, H, head_dim]
        """
        B, L, _ = x.shape

        # --- compress -> expand ---
        c_kv   = self.W_dkv(x)                                            # [B, L, C]
        k_base = self.W_uk(c_kv).view(B, L, self.num_heads, self.split_dim)
        k_r    = self.W_kr(c_kv).view(B, L, self.num_heads, self.rope_dim)
        v      = self.W_uv(c_kv).view(B, L, self.num_heads, self.head_dim)

        c_q    = self.W_dq(x)
        q_base = self.W_uq(c_q).view(B, L, self.num_heads, self.split_dim)
        q_r    = self.W_qr(c_q).view(B, L, self.num_heads, self.rope_dim)

        # --- apply RoPE only to the rotary chunks ---
        # (DeRoPE rotates the first rope_dim and concatenates the pass-through part)
        q = torch.cat([q_base, q_r], dim=-1)   # temp [B, L, H, head_dim]
        k = torch.cat([k_base, k_r], dim=-1)

        q = self.rope(q, seq_len=L,start_pos = seq_start)            # still [B, L, H, head_dim]
        k = self.rope(k, seq_len=L,start_pos = seq_start)

        # --- KV cache concat (after we have full head_dim keys) ---
        if past_kv is not None:
            k_past, v_past = past_kv          # [B, Lp, H, head_dim], [B, Lp, H, head_dim]
            k = torch.cat([k_past, k], dim=1) # [B, Lp+L, H, head_dim]
            v = torch.cat([v_past, v], dim=1) # [B, Lp+L, H, head_dim]

        # --- attention ---
        # q: [B, Lq, H, Dh], k: [B, Lk, H, Dh]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)

        if use_causal_mask:
            Lq, Lk = q.shape[1], k.shape[1]
            # Let queries correspond to the last Lq positions of the length-(Lk) sequence
            # Then mask everything above the diagonal offset by (Lk - Lq)
            offset = Lk - Lq
            causal = torch.triu(torch.ones(Lq, Lk, device=x.device, dtype=torch.bool), diagonal=1+offset)
            scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context
        cnt_x = torch.einsum("bhqk,bkhd->bqhd", attn, v).contiguous().view(B, -1, self.d_out)
        cnt_x = self.output(cnt_x)

        # cache full k and v (already head_dim each)
        new_cache = (k, v)
        return cnt_x, new_cache