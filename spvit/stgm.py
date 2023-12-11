from timm.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block
from timm.models.mlp_mixer import MixerBlock


class SGTMAttn(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0,
        proj_drop=0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__(
            dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer
        )
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x, semantic_tokens):
        B, N, C = x.shape
        Br, Nr, Cr = semantic_tokens.shape
        # Calculate Q matrix from registers
        q = (
            self.q(semantic_tokens)
            .reshape(Br, Nr, 1, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        # Calculate KV matrix from tokens
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q = q.unbind(0)[0]
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, Nr, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0,
        attn_drop=0,
        init_values=None,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        self.attn = SGTMAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x, semantic_tokens):
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        B, N, C = x.shape
        Bsem, Nsem, Csem = semantic_tokens.shape

        x = self.norm1(x)
        s_tokens = self.norm1(semantic_tokens)
        s_tokens = self.attn(x, s_tokens)
        s_tokens = semantic_tokens + self.drop_path1(self.ls1(s_tokens))
        s_tokens = s_tokens + self.drop_path2(self.ls2(self.mlp(self.norm2(s_tokens))))
        return s_tokens


class STGM(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0,
        attn_drop=0,
        init_values=None,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        window_size=4,
        num_patches=196,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.num_patches = num_patches
        self.num_semantic_tokens = window_size

        # self.s1_mixer = MixerBlock(dim, self.num_semantic_tokens)
        # self.s2_mixer = MixerBlock(dim, self.num_semantic_tokens)
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(self.num_patches, self.num_semantic_tokens),
        #     nn.GELU(),
        # )

        self.bottleneck = nn.Sequential(
            MixerBlock(dim, self.num_patches),
            nn.Conv1d(self.num_patches, self.num_semantic_tokens, 1),
            nn.GELU(),
            MixerBlock(dim, self.num_semantic_tokens),
        )

        self.block_1 = TransformerBlock(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        self.block_2 = TransformerBlock(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )

        self.global_center = nn.Parameter(
            torch.zeros((1, self.num_semantic_tokens, dim)),
            requires_grad=True,
        )

    def init_weights(self):
        nn.init.normal_(self.global_center, std=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int((N - 1) ** (0.5))

        # Computing S1
        # cls_token, x = x[:, :1], x[:, 1:]
        # x = x.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        # spatial_initial_center = F.adaptive_avg_pool2d(x, (self.window_size))
        # spatial_initial_center = spatial_initial_center.permute(0, 2, 3, 1).reshape(
        #     (B, -1, C)
        # )
        # spatial_initial_center = self.s1_mixer(spatial_initial_center)
        # x = x.permute(0, 2, 3, 1).reshape((B, -1, C))
        spatial_initial_center = self.bottleneck(x)

        # spatial_initial_center = torch.cat([cls_token, spatial_initial_center], dim=1)
        s1 = self.block_1(x, spatial_initial_center)

        # Computing S2
        x = torch.concat([x, s1], dim=1)
        s2 = self.block_2(x, s1 + self.global_center)

        return s2


if __name__ == "__main__":
    x = torch.randn((1, 197, 768))
    print(x.shape)

    window_size = 4
    B, N, C = x.shape
    # W = H = int(N ** (0.5))
    # x = x.reshape((B, H, W, C)).permute(0, 3, 1, 2)

    # xx = F.unfold(x, (6,6), stride=6)

    print(x.shape)
    # print(xx.reshape(B,C, 36, -1).shape)
    # print(F.adaptive_max_pool2d(x, (4, 4)).shape)

    stgm = STGM(768, 12, window_size=4)
    stgm(x)
