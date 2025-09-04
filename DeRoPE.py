
import torch
import torch.nn as nn

# We will use Decoupled Rotarary Positional encoding and also use compress kv vector to save memeroy and fast inference
#Decoupled Rotarty Positinal encoding

#Decoupled Rotarty Positinal encoding

# Function to make rotation of a 2d vector
def rotate_half(x):
  x1,x2 = x.chunk(2,dim=-1) # chunking a 2d vector with 2*2 matrix
  return torch.cat([-x2,x1],dim = -1) # minus sign ensure the rotation

# Another function to apply rotarary
def apply_rotary(x,cos,sin):
  # x is the input embedding (shape [batch, seq_len, hidden_dim]).cos and sin are precomputed rotation matrices (broadcastable tensors)
  # that have shape [batch, seq_len, rot_dim].cos.shape[-1] = rot_dim (the number of dimensions we apply RoPE to).
  #x_rot,x_pass = x.split(cos.shape[-1],dim = 1)
  x_rot = x[..., :cos.shape[-1]]   # first rope_dim dimensions
  x_pass = x[..., cos.shape[-1]:]

  # x_rot * cos will give (x1​,x2​)⋅cosθ=(x1​cosθ,x2​cosθ)
  # rotate_half(x_rot) * sin will give (−x2​,x1​)⋅sinθ=(−x2​sinθ,x1​sinθ)
  # by adding them we will get (x1​cosθ−x2​sinθ,x2​cosθ+x1​sinθ)

  x_rot = x_rot * cos + rotate_half(x_rot)*sin

  return torch.cat([x_rot,x_pass],dim = -1)

# DeRoPE class
class DeRoPE(nn.Module):
    def __init__(self, rope_dim, scale=40):
        super().__init__()
        assert rope_dim % 2 == 0, "rope_dim must be even for rotate_half()"
        self.rope_dim = rope_dim
        self.scale = scale

    def forward(self, x, seq_len=None, start_pos=0):
        """
        x: [B, L, H, D_r+]  (rotate first rope_dim)
        start_pos: absolute position offset for KV caching
        """
        if seq_len is None:
            seq_len = x.shape[1]

        device   = x.device
        half_dim = self.rope_dim // 2

        # angular frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t        = (torch.arange(seq_len, device=device).float() + start_pos) / self.scale  # add offset
        angles   = torch.einsum("l,d->ld", t, inv_freq)
        angles   = torch.cat([angles, angles], dim=-1)

        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, rope_dim]
        sin = angles.sin().unsqueeze(0).unsqueeze(2)

        x_rot  = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin

        return torch.cat([x_rot, x_pass], dim=-1)
