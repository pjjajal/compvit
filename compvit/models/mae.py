import logging
from functools import partial
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import SwiGLUFFNFused

from .compvit import CompViT
from timm.layers.weight_init import trunc_normal_

class MaskTokens(nn.Module):
    def __init__(self, decoder_embed_dim, num_patches) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(
            torch.zeros((1, 1, decoder_embed_dim)), requires_grad=True
        )

    def initialize_weights(self):
        # mask token init
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, num_patches, compressed_tokens):
        B = compressed_tokens.shape[0]
        mask_tokens = self.mask_token.repeat(B, num_patches, 1)
        mask_tokens = torch.cat([mask_tokens, compressed_tokens], dim=1)
        return mask_tokens

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, *args):
        x = x[:, 0]
        x = self.mlp(x)
        # x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

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
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
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
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x, num_tokens):
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return x_norm[:, :num_tokens, :]


class MAECompVit(nn.Module):
    def __init__(
        self,
        baseline,
        encoder: CompViT,
        decoder: Union[Decoder, nn.Identity],
        baseline_embed_dim,
        embed_dim,
        decoder_embed_dim,
        loss: Literal["l2"] = "l2",
        tradeoff: float = 5e-3,
        use_logit=True,
        baseline_head=None,
        encoder_head=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.baseline = baseline
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.tradeoff = tradeoff
        self.use_logit = use_logit

        self.num_patches = self.encoder.total_tokens
        self.num_compressed_tokens = self.encoder.num_compressed_tokens

        # Linear projection from encoder embeddings to decoder embeddings
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Linear projection from decoder embeddings to baseline embeddings
        self.decoder_pred = nn.Linear(decoder_embed_dim, baseline_embed_dim, bias=True)

        if isinstance(self.decoder, Decoder):
            self.mask_tokens = MaskTokens(decoder_embed_dim, self.num_patches)

        if self.use_logit:
            self.baseline_head = baseline_head
            self.encoder_head = encoder_head

        self.initialize_weights()

    def initialize_weights(self):
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
        parameters.extend(self.decoder.parameters())
        if isinstance(self.decoder, Decoder):
            parameters.extend(self.mask_tokens.parameters())
        
        return parameters

    @torch.no_grad()
    def forward_baseline(self, x):
        baseline_outputs = self.baseline.forward_features(x)
        # cls_token = baseline_outputs["x_norm_clstoken"]
        # patch_tokens = baseline_outputs["x_norm_patchtokens"]
        # baseline_outputs = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return baseline_outputs['x_norm']
    
    @torch.no_grad()
    def forward_baseline_head(self,x):
        pass

    def forward_encoder(self, x):
        encoder_outputs = self.encoder.forward_features(x)
        # cls_token = encoder_outputs["x_norm_clstoken"]
        # patch_tokens = encoder_outputs["x_norm_patchtokens"]
        # encoder_outputs = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return encoder_outputs['x_norm']
    
    def forward_encoder_head(self, x):
        pass

    def forward_decoder(self, encoder_outputs, N_baseline):
        encoder_outputs = self.decoder_embed(encoder_outputs)
        # Handle non-identify decoder
        if isinstance(self.decoder, Decoder):
            encoder_outputs = self.mask_tokens(N_baseline, encoder_outputs)
        decoder_outputs = self.decoder(encoder_outputs, N_baseline)
        return decoder_outputs

    def forward_loss(self, baseline_outputs, decoder_outputs):
        if self.loss == "ce" or isinstance(self.decoder, DINOHead):
            baseline_outputs = baseline_outputs[:, 0]
        # Project decoder embed dim to baseline embed dim
        decoder_outputs = self.decoder_pred(decoder_outputs)

        if self.loss == "l2":
            loss = self.l2_loss(baseline_outputs, decoder_outputs)
        elif self.loss == "ce":
            loss = self.ce_loss(baseline_outputs, decoder_outputs)
        elif self.loss == "smooth":
            loss = self.smooth_loss(baseline_outputs, decoder_outputs)
        return loss

    def l2_loss(self, baseline_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        # mean = baseline_outputs.mean(dim=-1, keepdim=True)
        # var = baseline_outputs.var(dim=-1, keepdim=True)
        # baseline_outputs = (baseline_outputs - mean) / (var + 1.e-6)**.5

        loss = (decoder_outputs - baseline_outputs) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.mean()
        return loss
        # return F.mse_loss(decoder_outputs, baseline_outputs, reduction="mean")
    
    def smooth_loss(self, baseline_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        return F.smooth_l1_loss(baseline_outputs, decoder_outputs)

    def ce_loss(self, baseline_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        baseline_outputs = F.softmax(baseline_outputs, dim=-1)
        loss = torch.sum(-baseline_outputs * F.log_softmax(decoder_outputs, dim=-1), dim=-1)
        loss = loss.mean()
        return loss

    def forward(self, x, xbaseline):
        baseline_outputs = self.forward_baseline(xbaseline)
        encoder_outputs = self.forward_encoder(x)

        _, N_baseline, _ = baseline_outputs.shape
        decoder_outputs = self.forward_decoder(encoder_outputs, N_baseline)

        loss = self.forward_loss(baseline_outputs, decoder_outputs)

        # Stupid hack to make multi-gpu work without issue for Lightning
        # all_params = torch.sum(torch.stack([torch.sum(p) for p in self.parameters()]))
        # loss = loss + 0 * all_params
        return loss
