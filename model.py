import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from NeedClasses import LayerNorm

# Hybrid Architecture
class ShomsherLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['dropout'])

        # Can't use Sequential, because we need per-layer past_kv
        self.t_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['num_layers'])])

        self.final_ln = LayerNorm(cfg['emb_dim'])
        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])

    def forward(self, x, return_cache=False, past_kv=None,seq_start=0):
        batch, seq_len = x.shape

        # Token embeddings only (RoPE handles positions)
        x = self.tok_emb(x)
        x = self.drop_emb(x)

        new_caches = []
        for i, block in enumerate(self.t_blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_cache = block(x, past_kv=layer_past,seq_start=seq_start)
            if return_cache:
                new_caches.append(layer_cache)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        if return_cache:
            return logits, new_caches
        else:
            return logits
