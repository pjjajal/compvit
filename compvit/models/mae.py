import torch
import torch.nn as nn
import torch.nn.functional as F
from .compvit import CompViT
from typing import Literal


class Mask(nn.Module):
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
        loss: Literal["l2", "barlow"] = "l2",
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

        # MAE decoder specifics
        # self.mask_token = nn.Parameter(
        #     torch.zeros((1, 1, decoder_embed_dim)), requires_grad=True
        # )
        # self.decoder_pos_embed = nn.Parameter(
        #     torch.zeros((1, self.num_patches, decoder_embed_dim)),
        #     requires_grad=True,
        # )
        # self.mask_gen = Mask(decoder_embed_dim, self.num_patches)

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

        # decoder norm init
        # nn.init.constant_(self.decoder_norm.bias, 0)
        # nn.init.constant_(self.decoder_norm.weight, 1.0)

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
        B,C = baseline_outputs.shape

        # Project decoder embed dim to baseline embed dim
        decoder_outputs = self.decoder_pred(decoder_outputs)

        if self.loss == "l2":
            loss = self.l2_loss(baseline_outputs, decoder_outputs)
        elif self.loss == "barlow":
            loss = self.barlow_loss(baseline_outputs, decoder_outputs)
        return loss

    def l2_loss(self, baseline_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        return F.smooth_l1_loss(decoder_outputs, baseline_outputs, reduction="mean")

    def forward(self, x, xbaseline):
        baseline_outputs = self.forward_baseline(xbaseline)
        encoder_outputs = self.forward_encoder(x)
        # decoder_outputs = self.forward_decoder(encoder_outputs)

        decoder_outputs = self.decoder_embed(encoder_outputs)

        loss = self.forward_loss(baseline_outputs, decoder_outputs)
        return loss
