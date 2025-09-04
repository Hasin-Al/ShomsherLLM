
import torch
import torch.nn as nn
from MultiHeadLatentAttention import MultiHeadLatentAttention
from NeedClasses import LayerNorm, RMSNorm, FeedForwardMoE


# Transformer block with Multi head latent attention
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mhattention = MultiHeadLatentAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['num_heads'],
            rope_dim=cfg['rope_dim'],
            compressed_dim=cfg['compressed_dim'],
            dropout=cfg['dropout'],
            bias_qkv=cfg['bias_qkv']
        )
        self.ln1 = LayerNorm(cfg['emb_dim'])
        self.ff = FeedForwardMoE(cfg)
        self.ln2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['dropout'])
        self.rmsnorm1 = RMSNorm(cfg['emb_dim'])
        self.rmsnorm2 = RMSNorm(cfg['emb_dim'])

    def forward(self, x, past_kv=None,seq_start=0):
        # Attention block
        shortcut = x
        x = self.ln1(x)
        x, updated_kv = self.mhattention(x, past_kv=past_kv,seq_start=seq_start)
        x = self.rmsnorm1(x)
        x = self.dropout(x)
        x = shortcut + x  # residual

        # Feed-forward block
        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.rmsnorm2(x)
        x = self.dropout(x)
        x = shortcut + x  # residual

        return x, updated_kv

