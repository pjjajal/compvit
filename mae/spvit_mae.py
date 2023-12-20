import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

import spvit
from spvit.spvit_model import SPViT
from timm.models.vision_transformer import Block


class MAEViT(nn.Module):
    def __init__(
        self,
        baseline,
        encoder: SPViT,
        decoder: nn.TransformerDecoder,
        baseline_embed_dim,
        embed_dim,
        decoder_embed_dim,
        norm_layer,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.baseline = baseline
        self.encoder = encoder
        self.decoder = decoder

        self.num_patches = (
            self.encoder.patch_embed.num_patches + self.encoder.num_prefix_tokens
        )
        self.num_semantic_tokens = self.encoder.stgm.num_semantic_tokens

        # MAE decoder specifics
        self.mask_token = nn.Parameter(
            torch.zeros((1, 1, decoder_embed_dim)), requires_grad=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros((1, self.num_patches, decoder_embed_dim)),
            requires_grad=True,
        )

        # Decoder Norm
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Linear projection from encoder embeddings to decoder embeddings
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Linear projection from decoder embeddings to baseline embeddings
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, baseline_embed_dim, bias=True
        )  # decoder to patch


        self.initialize_weights()

    def initialize_weights(self):
        # decoder position embedding initialization
        trunc_normal_(self.decoder_pos_embed, std=0.02)

        # mask token init
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # decoder_embed and decoder_pred initialization
        self._init_weights(self.decoder_embed)
        self._init_weights(self.decoder_pred)

        # decoder norm init
        nn.init.constant_(self.decoder_norm.bias, 0)
        nn.init.constant_(self.decoder_norm.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_parameters(self, whole=False, train_bottleneck = False, blocks=False):
        parameters = []

        if whole:
            parameters.extend(self.encoder.parameters())
        else:
            parameters.extend(self.encoder.peft_parameters(train_bottleneck=train_bottleneck, blocks=blocks))
        parameters.extend(self.decoder.parameters())
        parameters.extend(self.decoder_embed.parameters())
        parameters.extend(self.decoder_pred.parameters())
        parameters.extend(self.decoder_norm.parameters())
        # parameters.extend(self.mask_token)
        # parameters.extend(self.decoder_pos_embed)

        return parameters

    @torch.no_grad()
    def forward_baseline(self, x):
        baseline_outputs = self.baseline.forward_features(x)
        return baseline_outputs

    def forward_encoder(self, x):
        encoder_outputs = self.encoder.forward_features(x)
        return encoder_outputs

    def forward_decoder(self, encoder_outputs):
        B, _, _ = encoder_outputs.shape

        # Create mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_patches, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed

        # Project encoder output embedding dim to decoder
        encoder_outputs = self.decoder_embed(encoder_outputs)
        # Decode forward pass
        decoder_outputs = self.decoder(mask_tokens, encoder_outputs)
        decoder_outputs = self.decoder_norm(decoder_outputs)
        return decoder_outputs

    def forward_loss(self, baseline_outputs, decoder_outputs):
        B, N, C = baseline_outputs.shape

        # Project decoder embed dim to baseline embed dim
        decoder_outputs = self.decoder_pred(decoder_outputs)

        mean = baseline_outputs.mean(dim=-1, keepdim=True)
        var = baseline_outputs.var(dim=-1, keepdim=True)
        target = (baseline_outputs - mean) / (var + 1.e-6)**.5

        # L2 norm over dim
        loss = (baseline_outputs - decoder_outputs).norm(p=2, dim=-1)
        # Average over tokens
        loss = loss.mean(dim=-1)
        # Sum over batches
        loss = loss.mean()

        return loss

    def forward(self, x):
        baseline_outputs = self.forward_baseline(x)
        encoder_outputs = self.forward_encoder(x)
        decoder_outputs = self.forward_decoder(encoder_outputs)
        loss = self.forward_loss(baseline_outputs, decoder_outputs)
        return loss


def mae_factory(
    model="deit3_small_patch16_224",
    baseline_model="deit3_small_patch16_224",
    window_size=4,
    stgm_location=[5, 6],
    bottleneck=True,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.0,
    norm_layer=nn.LayerNorm,
):
    # This is the baseline model which we will distill from.
    baseline = timm.create_model(baseline_model, pretrained=True)

    # Encoder
    encoder = spvit.create_model(
        model,
        window_size=window_size,
        stgm_location=stgm_location,
        bottleneck=bottleneck,
        pretrained=False
    )

    # Decoder
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=decoder_embed_dim,
        nhead=decoder_num_heads,
        dim_feedforward=int(decoder_embed_dim * mlp_ratio),
        dropout=0.0,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=True,
    )
    decoder = nn.TransformerDecoder(decoder_layer, decoder_depth)

    # decoder = nn.Sequential(*[Block(
    #     dim=decoder_embed_dim,
    #     num_heads=decoder_num_heads,
    #     mlp_ratio=mlp_ratio,
    #     act_layer=nn.GELU,
    #     init_values=1e-6,
    # ) for i in range(decoder_depth)])

    return MAEViT(
        baseline,
        encoder,
        decoder,
        baseline.embed_dim,
        encoder.embed_dim,
        decoder_embed_dim,
        norm_layer,
    )


if __name__ == "__main__":
    mae = mae_factory()
    print(mae(torch.randn(2, 3, 224, 224)))