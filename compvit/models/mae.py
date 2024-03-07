import logging
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import SwiGLUFFNFused

from .compvit import CompViT


class MaskTokens(nn.Module):
    def __init__(self, decoder_embed_dim, num_patches) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(
            torch.zeros((1, 1, decoder_embed_dim)), requires_grad=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros((1, num_patches, decoder_embed_dim)),
            requires_grad=True,
        )

    def initialize_weights(self):
        # decoder position embedding initialization
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # mask token init
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, B, num_patches):
        mask_tokens = self.mask_token.repeat(B, num_patches, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed
        return mask_tokens


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        init_values=None,  # for layerscale: None or 0 => no layerscale
        num_tokens=1,
    ):
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = num_tokens
        self.n_blocks = depth
        self.num_heads = num_heads

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return x_norm[:self.num_tokens]


class MAECompVit(nn.Module):
    def __init__(
        self,
        baseline,
        encoder: CompViT,
        decoder: nn.TransformerDecoder,
        baseline_embed_dim,
        embed_dim,
        decoder_embed_dim,
        norm_layer,
        loss: Literal["l2"] = "l2",
        tradeoff: float = 5e-3,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.baseline = baseline
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.tradeoff = tradeoff

        self.num_patches = self.encoder.total_tokens
        self.num_compressed_tokens = self.encoder.num_compressed_tokens

        # Decoder Norm
        # self.decoder_norm = norm_layer(decoder_embed_dim)

        # Linear projection from encoder embeddings to decoder embeddings
        self.decoder_embed = nn.Linear(embed_dim * 2, decoder_embed_dim * 2, bias=True)
        # Linear projection from decoder embeddings to baseline embeddings
        self.decoder_pred = nn.Linear(
            decoder_embed_dim * 2, baseline_embed_dim * 2, bias=True
        )  # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # decoder position embedding initialization
        # torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # mask token init
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # self.mask_gen.initialize_weights()

        # decoder_embed and decoder_pred initialization
        self._init_weights(self.decoder_embed)
        self._init_weights(self.decoder_pred)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_parameters(self, whole=False, train_bottleneck=False, blocks=False):
        parameters = []

        if whole:
            parameters.extend(self.encoder.parameters())
        else:
            parameters.extend(
                self.encoder.peft_parameters(
                    train_bottleneck=train_bottleneck, blocks=blocks
                )
            )

        parameters.extend(self.decoder_embed.parameters())
        parameters.extend(self.decoder_pred.parameters())
        # parameters.extend(self.decoder.parameters())
        # parameters.extend(self.decoder_norm.parameters())
        # parameters.extend(self.mask_gen.parameters())
        # parameters.extend(self.mask_token)
        # parameters.extend(self.decoder_pos_embed)

        return parameters

    @torch.no_grad()
    def forward_baseline(self, x):
        baseline_outputs = self.baseline.forward_features(x)
        cls_token = baseline_outputs["x_norm_clstoken"]
        patch_tokens = baseline_outputs["x_norm_patchtokens"]
        baseline_outputs = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return baseline_outputs

    def forward_encoder(self, x):
        encoder_outputs = self.encoder.forward_features(x)
        cls_token = encoder_outputs["x_norm_clstoken"]
        patch_tokens = encoder_outputs["x_norm_patchtokens"]
        encoder_outputs = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return encoder_outputs

    def forward_decoder(self, encoder_outputs):
        B, _, _ = encoder_outputs.shape

        # Create mask tokens
        # mask_tokens = self.mask_token.repeat(B, self.num_patches, 1)
        # mask_tokens = mask_tokens + self.decoder_pos_embed
        mask_tokens = self.mask_gen(B, self.num_patches)

        # Project encoder output embedding dim to decoder
        encoder_outputs = self.decoder_embed(encoder_outputs)
        # Decode forward pass
        decoder_outputs = self.decoder(mask_tokens, encoder_outputs)
        decoder_outputs = self.decoder_norm(decoder_outputs)
        return decoder_outputs

    def forward_loss(self, baseline_outputs, decoder_outputs):
        # B, N, C = baseline_outputs.shape
        B, C = baseline_outputs.shape

        # Project decoder embed dim to baseline embed dim
        decoder_outputs = self.decoder_pred(decoder_outputs)

        if self.loss == "l2":
            loss = self.l2_loss(baseline_outputs, decoder_outputs)
        return loss

    def l2_loss(self, baseline_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        return F.mse_loss(decoder_outputs, baseline_outputs, reduction="mean")

    def forward(self, x, xbaseline):
        baseline_outputs = self.forward_baseline(xbaseline)
        encoder_outputs = self.forward_encoder(x)
        # decoder_outputs = self.forward_decoder(encoder_outputs)

        decoder_outputs = self.decoder_embed(encoder_outputs)
        loss = self.forward_loss(baseline_outputs, decoder_outputs)

        # Stupid hack to make multi-gpu work without issue for Lightning
        all_params = torch.sum(torch.stack([torch.sum(p) for p in self.parameters()]))
        loss = loss + 0 * all_params
        return loss
