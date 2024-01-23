import torch
from torch import Tensor
from torch import nn

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, compressed_tokens: Tensor, get_attn=False) -> Tensor:
        B, N, C = x.shape
        Br, Nr, Cr = compressed_tokens.shape

        q = self.q(compressed_tokens).reshape(Br, Nr, 1, self.num_heads, Cr // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = q[0] * self.scale, kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nr, Cr)
        x = self.proj(x)
        x = self.proj_drop(x)

        if get_attn:
            return x, attn

        return x, None
    
if __name__ == "__main__":
    attn = CrossAttention(384, 8, True)

    out = attn(torch.randn((1, 196, 384)), torch.randn((1, 8, 384)))
    print(out.shape)