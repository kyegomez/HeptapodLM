import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


# 2D Rotary Embedding
class Rotary2DEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, height, width, *, device):
        seq_h = torch.arange(height, device=device, dtype=self.inv_freq.dtype)
        seq_w = torch.arange(width, device=device, dtype=self.inv_freq.dtype)
        freqs_h = einsum("i , j -> i j", seq_h, self.inv_freq)
        freqs_w = einsum("i , j -> i j", seq_w, self.inv_freq)
        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        return freqs


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


# Local 2D Attention
class Local2DAttention(nn.Module):
    def __init__(self, dim, window_size=1):
        super().__init__()
        self.window_size = window_size
        self.scale = dim**-0.5

    def forward(self, matrix):
        B, H, W, D = matrix.shape
        local_contexts = []

        for i in range(H):
            for j in range(W):
                # Extract local window
                i_start, i_end = max(0, i - self.window_size), min(
                    H, i + self.window_size + 1
                )
                j_start, j_end = max(0, j - self.window_size), min(
                    W, j + self.window_size + 1
                )
                local_window = matrix[:, i_start:i_end, j_start:j_end, :].reshape(
                    B, -1, D
                )

                q = matrix[:, i, j, :].unsqueeze(1)
                k, v = local_window, local_window

                # Attention
                attn_scores = torch.bmm(q * self.scale, k.transpose(1, 2))
                attn_probs = F.softmax(attn_scores, dim=-1)
                context = torch.bmm(attn_probs, v).squeeze(1)
                local_contexts.append(context)

        local_contexts = torch.stack(local_contexts, dim=1).view(B, H, W, D)
        return local_contexts


# Non-Linear Transformer Block
class NonLinearTransformerBlock(nn.Module):
    def __init__(self, dim, window_size=1):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.local_attn = Local2DAttention(dim, window_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        attn_out = self.local_attn(self.norm(x))
        x = x + attn_out
        return x + self.mlp(self.norm(x))


# Non-Linear Transformer Main Model
class NonLinearTransformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, matrix_dim, window_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [NonLinearTransformerBlock(dim) for _ in range(depth)]
        )
        self.rotary_emb = Rotary2DEmbedding(dim)
        self.local_attn = Local2DAttention(dim, window_size)
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, matrix):
        b, h, w = matrix.size()
        matrix = self.embedding(matrix)
        pos_emb = self.rotary_emb(h, w, device=matrix.device)

        for block in self.blocks:
            matrix = matrix + block(matrix)
            matrix = matrix + self.local_attn(matrix)

        matrix = matrix + pos_emb
        logits = self.to_logits(matrix)
        return logits

