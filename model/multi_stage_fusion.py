import torch
import torch.nn as nn

from model.token_compressor import TokenCompressor


class SharedTokenCompressor(nn.Module):
    """Compatibility container for the shared Stage-4 token compressor."""

    def __init__(
        self,
        hidden_dim=512,
        num_heads=8,
        image_feature_dims=None,
        num_compressed_tokens=4,
    ):
        super().__init__()
        if image_feature_dims is None or len(image_feature_dims) != 1:
            raise ValueError(
                "SharedTokenCompressor expects exactly one Stage-4 feature dimension"
            )

        self.compressor = TokenCompressor(
            feature_dim=image_feature_dims[0],
            token_dim=hidden_dim,
            num_tokens=num_compressed_tokens,
            num_heads=num_heads,
            orthogonal_init=True,
            )

    def compute_orthogonality_loss(self):
        """Return the Stage-4 compressor's orthogonal token regularizer."""
        loss = torch.tensor(
            0.0,
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        loss = loss + self.compressor.compute_orthogonality_loss()
        return loss