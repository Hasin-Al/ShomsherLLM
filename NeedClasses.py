<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalization class
class LayerNorm(nn.Module):
  def __init__(self,embed_dim):
    super().__init__()
    self.eps = 1e-6
    self.scale = nn.Parameter(torch.ones(embed_dim))
    self.shift = nn.Parameter(torch.zeros(embed_dim))

  def forward(self,x):
    mean = x.mean(dim=-1,keepdim = True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x-mean)/torch.sqrt(var+self.eps)
    return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))  # only scale, no shift

    def forward(self, x):
        # Compute RMS over last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        return self.scale * norm_x


# Activation function
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))

# MoE Layer

# Expert (like your FFN but smaller)
class Expert(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            SwiGLU(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.ffn(x)


# Mixture of Experts
class MoE(nn.Module):
    def __init__(self, emb_dim, num_experts=4, hidden_dim=None, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_dim = hidden_dim or 4 * emb_dim

        # Create experts
        self.experts = nn.ModuleList([Expert(emb_dim, hidden_dim) for _ in range(num_experts)])

        # Gating network
        self.gate = nn.Linear(emb_dim, num_experts)

    def forward(self, x):
        # x: [batch, seq, emb]
        batch_size, seq_len, emb_dim = x.shape

        # Gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)   # [batch, seq, num_experts]

        # Select top-k experts
        topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # both [batch, seq, k]

        # Normalize top-k scores
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x)

        for i in range(self.k):
            expert_idx = topk_idx[..., i]   # [batch, seq]
            expert_score = topk_scores[..., i].unsqueeze(-1)  # [batch, seq, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e).float().unsqueeze(-1)  # [batch, seq, 1]
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e](x)  # [batch, seq, emb]
                output += expert_out * expert_score * mask

        return output


# Replace your FeedForward with MoE
class FeedForwardMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.moe = MoE(cfg['emb_dim'], num_experts=4, hidden_dim=4*cfg['emb_dim'], k=1)

    def forward(self, x):
        return self.moe(x)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalization class
class LayerNorm(nn.Module):
  def __init__(self,embed_dim):
    super().__init__()
    self.eps = 1e-6
    self.scale = nn.Parameter(torch.ones(embed_dim))
    self.shift = nn.Parameter(torch.zeros(embed_dim))

  def forward(self,x):
    mean = x.mean(dim=-1,keepdim = True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x-mean)/torch.sqrt(var+self.eps)
    return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))  # only scale, no shift

    def forward(self, x):
        # Compute RMS over last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        return self.scale * norm_x


# Activation function
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))

# MoE Layer

# Expert (like your FFN but smaller)
class Expert(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            SwiGLU(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.ffn(x)


# Mixture of Experts
class MoE(nn.Module):
    def __init__(self, emb_dim, num_experts=4, hidden_dim=None, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_dim = hidden_dim or 4 * emb_dim

        # Create experts
        self.experts = nn.ModuleList([Expert(emb_dim, hidden_dim) for _ in range(num_experts)])

        # Gating network
        self.gate = nn.Linear(emb_dim, num_experts)

    def forward(self, x):
        # x: [batch, seq, emb]
        batch_size, seq_len, emb_dim = x.shape

        # Gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)   # [batch, seq, num_experts]

        # Select top-k experts
        topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # both [batch, seq, k]

        # Normalize top-k scores
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x)

        for i in range(self.k):
            expert_idx = topk_idx[..., i]   # [batch, seq]
            expert_score = topk_scores[..., i].unsqueeze(-1)  # [batch, seq, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e).float().unsqueeze(-1)  # [batch, seq, 1]
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e](x)  # [batch, seq, emb]
                output += expert_out * expert_score * mask

        return output


# Replace your FeedForward with MoE
class FeedForwardMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.moe = MoE(cfg['emb_dim'], num_experts=4, hidden_dim=4*cfg['emb_dim'], k=1)

    def forward(self, x):
        return self.moe(x)
>>>>>>> b2a3c63f3e93c5bb09be358e3702f1e3bcc310d1
