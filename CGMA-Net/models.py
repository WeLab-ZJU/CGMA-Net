from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.feature_dim = embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out", nonlinearity="relu")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RETFoundCFPEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=dropout,
                    attn_drop=0.0,
                    norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
                )
                for _ in range(depth)
            ]
        )
        self.global_pool = True
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.feature_dim = embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.size(0), -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=False)
            x = self.fc_norm(x)
            return x
        x = self.norm(x)
        return x[:, 0]


def interpolate_pos_embed(model: RETFoundCFPEncoder, checkpoint_model: Dict[str, torch.Tensor]) -> None:
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size == new_size:
        return
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_retfound_pretrained(encoder: RETFoundCFPEncoder, checkpoint_path: str) -> RETFoundCFPEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    checkpoint_model = {}
    for key, value in state_dict.items():
        key = key.replace("backbone.", "")
        key = key.replace("mlp.w12.", "mlp.fc1.")
        key = key.replace("mlp.w3.", "mlp.fc2.")
        if key.startswith("decoder_") or key.startswith("head.") or key == "mask_token":
            continue
        checkpoint_model[key] = value
    state_dict_current = encoder.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and key in state_dict_current and checkpoint_model[key].shape != state_dict_current[key].shape:
            del checkpoint_model[key]
    interpolate_pos_embed(encoder, checkpoint_model)
    encoder.load_state_dict(checkpoint_model, strict=False)
    return encoder


class EncoderHead(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        encoder_out_dim = getattr(encoder, "feature_dim", 512)
        self.proj = nn.Sequential(
            nn.Linear(encoder_out_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        projected = self.proj(features)
        return projected


class SingleModalityClassifier(nn.Module):
    def __init__(self, encoder_head: EncoderHead, feature_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder_head = encoder_head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder_head(x)
        logit = self.classifier(features).squeeze(1)
        return {"logit": logit, "features": features}


class CrossAttentionGateFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_tokens: int = 8,
        token_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.cfp_tokenizer = nn.Linear(feature_dim, num_tokens * token_dim)
        self.vf_tokenizer = nn.Linear(feature_dim, num_tokens * token_dim)
        self.cfp_to_vf_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vf_to_cfp_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cfp_reproject = nn.Sequential(
            nn.Linear(num_tokens * token_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.vf_reproject = nn.Sequential(
            nn.Linear(num_tokens * token_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _to_tokens(self, x: torch.Tensor, tokenizer: nn.Module) -> torch.Tensor:
        tokens = tokenizer(x)
        return tokens.view(x.size(0), self.num_tokens, self.token_dim)

    def forward(self, f_c: torch.Tensor, f_v: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        c_tokens = self._to_tokens(f_c, self.cfp_tokenizer)
        v_tokens = self._to_tokens(f_v, self.vf_tokenizer)

        c_from_v, _ = self.cfp_to_vf_attn(query=c_tokens, key=v_tokens, value=v_tokens, need_weights=False)
        v_from_c, _ = self.vf_to_cfp_attn(query=v_tokens, key=c_tokens, value=c_tokens, need_weights=False)

        c_from_v = c_from_v * mask[:, 1].view(-1, 1, 1)
        v_from_c = v_from_c * mask[:, 0].view(-1, 1, 1)

        h_c = f_c + self.cfp_reproject((c_tokens + c_from_v).reshape(f_c.size(0), -1))
        h_v = f_v + self.vf_reproject((v_tokens + v_from_c).reshape(f_v.size(0), -1))

        gate_input = torch.cat([h_c, h_v, mask], dim=1)
        gate_logits = self.gate(gate_input)
        masked_logits = gate_logits.masked_fill(mask <= 0, float("-inf"))
        both_missing = mask.sum(dim=1, keepdim=True) == 0
        if both_missing.any():
            fallback = torch.zeros_like(masked_logits)
            masked_logits = torch.where(both_missing.expand_as(masked_logits), fallback, masked_logits)

        beta = torch.softmax(masked_logits, dim=1)
        beta = beta * mask
        beta = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-8)

        z = beta[:, :1] * h_c + beta[:, 1:2] * h_v
        paired_mask = (mask.sum(dim=1) == 2).float()
        gate_entropy = -(beta * torch.log(beta.clamp_min(1e-8))).sum(dim=1)
        return {
            "fused": z,
            "cfp_updated": h_c,
            "vf_updated": h_v,
            "gate_weights": beta,
            "gate_logits": gate_logits,
            "gate_entropy": gate_entropy,
            "paired_mask": paired_mask,
        }


class MultiModalGatedClassifier(nn.Module):
    def __init__(
        self,
        cfp_encoder_head: EncoderHead,
        vf_encoder_head: EncoderHead,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cfp_encoder_head = cfp_encoder_head
        self.vf_encoder_head = vf_encoder_head
        self.fusion = CrossAttentionGateFusion(feature_dim=feature_dim, hidden_dim=128, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, cfp: torch.Tensor, vf: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfp_features = self.cfp_encoder_head(cfp)
        vf_features = self.vf_encoder_head(vf)
        fusion_output = self.fusion(cfp_features, vf_features, mask)
        fused = fusion_output["fused"]
        fused_with_mask = torch.cat([fused, mask], dim=1)
        logit = self.classifier(fused_with_mask).squeeze(1)
        uncertainty_raw = self.uncertainty_head(fused_with_mask).squeeze(1)
        uncertainty = torch.nn.functional.softplus(uncertainty_raw)
        return {
            "logit": logit,
            "uncertainty": uncertainty,
            "fused_features": fused,
            "cfp_features": cfp_features,
            "vf_features": vf_features,
            "cfp_updated_features": fusion_output["cfp_updated"],
            "vf_updated_features": fusion_output["vf_updated"],
            "gate_weights": fusion_output["gate_weights"],
            "gate_logits": fusion_output["gate_logits"],
            "gate_entropy": fusion_output["gate_entropy"],
            "paired_mask": fusion_output["paired_mask"],
        }


def build_cfp_encoder_head(retfound_checkpoint_path: str, feature_dim: int = 256, dropout: float = 0.3) -> EncoderHead:
    encoder = RETFoundCFPEncoder()
    encoder = load_retfound_pretrained(encoder, retfound_checkpoint_path)
    return EncoderHead(encoder=encoder, feature_dim=feature_dim, dropout=dropout)


def build_vf_encoder_head(
    feature_dim: int = 256,
    dropout: float = 0.3,
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    encoder_dropout: float = 0.1,
) -> EncoderHead:
    encoder = PatchTransformerEncoder(
        in_channels=1,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=encoder_dropout,
    )
    return EncoderHead(encoder=encoder, feature_dim=feature_dim, dropout=dropout)


def _extract_encoder_head_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "encoder_head."
    trimmed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            trimmed_state_dict[key[len(prefix):]] = value
    return trimmed_state_dict if trimmed_state_dict else state_dict


def load_cfp_encoder_head_checkpoint(
    checkpoint_path: str,
    retfound_checkpoint_path: str,
    feature_dim: int = 256,
    dropout: float = 0.3,
) -> EncoderHead:
    encoder_head = build_cfp_encoder_head(retfound_checkpoint_path=retfound_checkpoint_path, feature_dim=feature_dim, dropout=dropout)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    encoder_head.load_state_dict(_extract_encoder_head_state_dict(state_dict), strict=True)
    return encoder_head


def load_vf_encoder_head_checkpoint(
    checkpoint_path: str,
    feature_dim: int = 256,
    dropout: float = 0.3,
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    encoder_dropout: float = 0.1,
) -> EncoderHead:
    encoder_head = build_vf_encoder_head(
        feature_dim=feature_dim,
        dropout=dropout,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        encoder_dropout=encoder_dropout,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    encoder_head.load_state_dict(_extract_encoder_head_state_dict(state_dict), strict=True)
    return encoder_head
