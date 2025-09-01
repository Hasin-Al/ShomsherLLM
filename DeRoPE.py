<<<<<<< HEAD
import torch
import torch.nn as nn

# We will use Decoupled Rotarary Positional encoding and also use compress kv vector to save memeroy and fast inference
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

    def forward(self, x, seq_len=None,start_pos=0):
        """
        x: [B, L, H, D_r+]  (we will only rotate the first rope_dim along the last dim)
        If x's last dim == rope_dim, that's fine too.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        device   = x.device
        half_dim = self.rope_dim // 2

        # angular frequencies
        # calcualting the frequency scale # our main equation is θ(p,i)=p⋅1/10000**i/(d/2)
        # so inv_freq will be only 1/10000**i/(d/2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t        = (torch.arange(seq_len, device=device).float() + start_pos)/ self.scale     # [L]
        angles   = torch.einsum("l,d->ld", t, inv_freq)                          # [L, half_dim]
        angles   = torch.cat([angles, angles], dim=-1)                           # [L, rope_dim]

        # Broadcast to [1, L, 1, rope_dim] then expand across heads later
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, rope_dim]
        sin = angles.sin().unsqueeze(0).unsqueeze(2)  # [1, L, 1, rope_dim]

        # split x into (rotary part | pass-through part)
        x_rot  = x[..., :self.rope_dim]      # [B, L, H, rope_dim]
        x_pass = x[..., self.rope_dim:]      # [B, L, H, ?]

        # broadcast cos/sin across heads automatically (H dimension)
        x_rot = x_rot * cos + rotate_half(x_rot) * sin

=======
import torch
import torch.nn as nn

# We will use Decoupled Rotarary Positional encoding and also use compress kv vector to save memeroy and fast inference
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

    def forward(self, x, seq_len=None,start_pos=0):
        """
        x: [B, L, H, D_r+]  (we will only rotate the first rope_dim along the last dim)
        If x's last dim == rope_dim, that's fine too.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        device   = x.device
        half_dim = self.rope_dim // 2

        # angular frequencies
        # calcualting the frequency scale # our main equation is θ(p,i)=p⋅1/10000**i/(d/2)
        # so inv_freq will be only 1/10000**i/(d/2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t        = (torch.arange(seq_len, device=device).float() + start_pos)/ self.scale     # [L]
        angles   = torch.einsum("l,d->ld", t, inv_freq)                          # [L, half_dim]
        angles   = torch.cat([angles, angles], dim=-1)                           # [L, rope_dim]

        # Broadcast to [1, L, 1, rope_dim] then expand across heads later
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, rope_dim]
        sin = angles.sin().unsqueeze(0).unsqueeze(2)  # [1, L, 1, rope_dim]

        # split x into (rotary part | pass-through part)
        x_rot  = x[..., :self.rope_dim]      # [B, L, H, rope_dim]
        x_pass = x[..., self.rope_dim:]      # [B, L, H, ?]

        # broadcast cos/sin across heads automatically (H dimension)
        x_rot = x_rot * cos + rotate_half(x_rot) * sin

>>>>>>> b2a3c63f3e93c5bb09be358e3702f1e3bcc310d1
        return torch.cat([x_rot, x_pass], dim=-1)