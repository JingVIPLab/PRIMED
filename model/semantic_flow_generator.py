import torch
import torch.nn as nn


class SemanticFlowGenerator(nn.Module):
    """Generate semantic flow from CLAP audio features and DistilRoBERTa text features."""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 1024,
        common_dim: int = 512,
        num_heads: int = 8,
        beta: float = 0.5,
        disable_cached_memory: bool = False,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.beta = beta
        self.disable_cached_memory = disable_cached_memory

        self.text_proj = nn.Linear(text_dim, common_dim)
        self.audio_proj = nn.Linear(audio_dim, common_dim)

        self.self_attn = nn.MultiheadAttention(common_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(common_dim)

        self.ffn = nn.Sequential(
            nn.Linear(common_dim, common_dim * 4),
            nn.GELU(),
            nn.Linear(common_dim * 4, common_dim),
        )
        self.ffn_norm = nn.LayerNorm(common_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            text_features: [B, L, text_dim].
            audio_features: [B, T_audio, audio_dim].
            text_attention_mask: Optional [B, L] mask where True/1 marks valid text tokens.

        Returns:
            Semantic flow tensor with shape [B, L + T_audio, common_dim].
        """
        B, L, _ = text_features.shape

        text_proj = self.text_proj(text_features)
        audio_proj = self.audio_proj(audio_features)
        combined = torch.cat([text_proj, audio_proj], dim=1)

        combined_valid_mask = None
        key_padding_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.shape != (B, L):
                raise ValueError(
                    f"text_attention_mask must have shape {(B, L)}, "
                    f"got {tuple(text_attention_mask.shape)}"
                )
            text_valid_mask = text_attention_mask.to(device=combined.device, dtype=torch.bool)
            audio_valid_mask = torch.ones(
                B, audio_features.shape[1], device=combined.device, dtype=torch.bool
            )
            combined_valid_mask = torch.cat([text_valid_mask, audio_valid_mask], dim=1)
            key_padding_mask = ~combined_valid_mask

        attn_out, _ = self.self_attn(
            combined, combined, combined, key_padding_mask=key_padding_mask
        )
        combined = self.attn_norm(combined + attn_out)

        ffn_out = self.ffn(combined)
        combined = self.ffn_norm(combined + ffn_out)
        if combined_valid_mask is not None:
            combined = combined.masked_fill(~combined_valid_mask.unsqueeze(-1), 0.0)

        text_out = combined[:, :L, :]
        audio_out = combined[:, L:, :]
        audio_out = audio_out if self.disable_cached_memory else self._apply_cached_memory(audio_out)

        return torch.cat([text_out, audio_out], dim=1)

    def _apply_cached_memory(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Apply the cached-memory enhancement to temporal audio tokens."""
        _, T_audio, _ = audio_features.shape

        cumsum = torch.cumsum(audio_features, dim=1)
        prev_cumsum = torch.zeros_like(cumsum)
        prev_cumsum[:, 1:, :] = cumsum[:, :-1, :]

        counts = torch.arange(T_audio, device=audio_features.device, dtype=audio_features.dtype)
        counts = counts.unsqueeze(0).unsqueeze(-1)
        cached_memory = prev_cumsum / counts.clamp(min=1)

        return (self.beta + 1) * audio_features - self.beta * cached_memory
