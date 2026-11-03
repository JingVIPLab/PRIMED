import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityPriorProjector(nn.Module):
    """Project visual, text, and audio features into one CLIP-like prior space."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int = 768,
        audio_dim: int = 1024,
        semantic_dim: int = 512,
    ):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.similarity_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
    ):
        global_text_feature = text_features[:, 0:1, :]
        global_audio_feature = audio_features.mean(dim=1, keepdim=True)

        projected_visual = F.normalize(self.visual_proj(visual_features), dim=-1)
        projected_text = F.normalize(self.text_proj(global_text_feature), dim=-1)
        projected_audio = F.normalize(self.audio_proj(global_audio_feature), dim=-1)
        similarity_scale = self.similarity_scale.exp().clamp(max=100.0)

        return projected_visual, projected_text, projected_audio, similarity_scale
