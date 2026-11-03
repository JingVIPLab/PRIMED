import torch
import torch.nn as nn


class ModalityPriorDecoder(nn.Module):
    """Decode a language-guided audio/visual/joint modality prior."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, global_text_feature: torch.Tensor) -> torch.Tensor:
        return self.decoder(global_text_feature)
