# model.py
from typing import Optional
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embedding_dim)
        x = self.patch_embed(x)                      # (B, embed, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)             # (B, num_patches, embedding_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # multi-head self-attention with residual
        residual_1 = x
        x = self.norm1(x)
        attn_out = self.attn(x, x, x)[0]
        x = residual_1 + attn_out

        residual_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual_2
        return x


class MLPHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 1,
        embedding_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        num_classes: int = 10,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )

        mlp_hidden_dim = mlp_hidden_dim or (embedding_dim * 2)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoder(embedding_dim, num_heads, mlp_hidden_dim) for _ in range(num_layers)]
        )

        self.mlp_head = MLPHead(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embedding(x)  # (B, num_patches, embedding_dim)
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embedding_dim)
        x = x + self.pos_embedding

        for blk in self.transformer_blocks:
            x = blk(x)

        cls_out = x[:, 0]  # take cls token
        out = self.mlp_head(cls_out)
        return out
