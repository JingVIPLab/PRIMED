import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAwareSemanticAlignmentLoss(nn.Module):
    """Align the semantic flow with the foreground prototype and repel hard background tokens."""

    def __init__(self, temperature=0.07, target_size=64):
        super().__init__()
        self.temperature = temperature
        self.target_size = target_size
        self.visual_projection = nn.Linear(256, 512)

    def forward(
        self,
        fpn_features,
        gt_masks,
        semantic_flow,
        top_k=10,
        semantic_valid_mask=None,
    ):
        """
        Args:
            fpn_features: [B*T, 256, H3, W3] FPN features.
            gt_masks: [B*T, H, W] or [B*T, 1, H, W] ground-truth masks.
            semantic_flow: [B*T, N, 512] semantic tokens.
            top_k: Number of hard negative background tokens per sample.
            semantic_valid_mask: Optional [B*T, N] mask for valid semantic tokens.
        """
        B_T, C, _, _ = fpn_features.shape
        _, _, dim_semantic = semantic_flow.shape

        if gt_masks.dim() == 4 and gt_masks.shape[1] == 1:
            gt_masks = gt_masks.squeeze(1)

        gt_masks_64x64 = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=(self.target_size, self.target_size),
            mode='nearest',
        ).squeeze(1)
        fpn_features_64x64 = F.interpolate(
            fpn_features,
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False,
        )

        foreground_mask = (gt_masks_64x64 > 0.5).float()
        background_mask = (gt_masks_64x64 <= 0.5).float()

        fpn_flat = fpn_features_64x64.view(B_T, C, -1).transpose(1, 2)
        foreground_mask_flat = foreground_mask.view(B_T, -1)
        background_mask_flat = background_mask.view(B_T, -1)

        foreground_count_raw = foreground_mask_flat.sum(dim=1, keepdim=True)
        valid_foreground = foreground_count_raw.squeeze(-1) > 0
        foreground_count = foreground_count_raw.clamp(min=1)

        foreground_prototype = (fpn_flat * foreground_mask_flat.unsqueeze(-1)).sum(dim=1) / foreground_count
        projected_foreground_prototype = self.visual_projection(foreground_prototype)
        if semantic_valid_mask is None:
            semantic_anchor = semantic_flow.mean(dim=1)
        else:
            if semantic_valid_mask.shape != semantic_flow.shape[:2]:
                raise ValueError(
                    "semantic_valid_mask must match the first two dimensions of "
                    f"semantic_flow, got {tuple(semantic_valid_mask.shape)} "
                    f"and {tuple(semantic_flow.shape[:2])}"
                )
            semantic_weights = semantic_valid_mask.to(
                device=semantic_flow.device,
                dtype=semantic_flow.dtype,
            ).unsqueeze(-1)
            semantic_anchor = (semantic_flow * semantic_weights).sum(dim=1)
            semantic_anchor = semantic_anchor / semantic_weights.sum(dim=1).clamp(min=1.0)

        if not torch.any(valid_foreground):
            zero_loss = (projected_foreground_prototype.sum() + semantic_anchor.sum()) * 0.0
            return zero_loss, {
                'num_foreground_pixels': 0,
                'num_background_pixels': background_mask_flat.sum().item(),
                'num_positive_samples': 0,
                'num_hard_background_tokens': 0,
            }

        hard_background_tokens = self._select_hard_background_tokens(
            fpn_flat=fpn_flat,
            background_mask_flat=background_mask_flat,
            semantic_anchor=semantic_anchor,
            top_k=top_k,
            dim_semantic=dim_semantic,
        )

        normalized_anchor = F.normalize(semantic_anchor, p=2, dim=-1)
        normalized_positive = F.normalize(projected_foreground_prototype, p=2, dim=-1)
        normalized_negatives = F.normalize(hard_background_tokens, p=2, dim=-1)

        positive_similarity = (normalized_anchor * normalized_positive).sum(dim=-1)
        negative_similarity = torch.bmm(
            normalized_anchor.unsqueeze(1),
            normalized_negatives.transpose(1, 2),
        ).squeeze(1)

        positive_score = torch.exp(positive_similarity / self.temperature)
        negative_scores = torch.exp(negative_similarity / self.temperature)
        denominator = positive_score + negative_scores.sum(dim=1)
        sasa_loss = -torch.log(positive_score / denominator + 1e-8)
        sasa_loss = sasa_loss[valid_foreground].mean()

        diagnostics = {
            'num_foreground_pixels': foreground_count_raw.sum().item(),
            'num_background_pixels': background_mask_flat.sum(dim=1).sum().item(),
            'num_positive_samples': valid_foreground.sum().item(),
            'num_hard_background_tokens': hard_background_tokens.shape[1],
        }
        return sasa_loss, diagnostics

    def _select_hard_background_tokens(
        self,
        fpn_flat,
        background_mask_flat,
        semantic_anchor,
        top_k,
        dim_semantic,
    ):
        """Select the background tokens most similar to the global semantic anchor."""
        B_T = fpn_flat.shape[0]
        actual_k = min(top_k, self.target_size * self.target_size)
        hard_background_token_sets = []

        for i in range(B_T):
            bg_tokens_i = fpn_flat[i][background_mask_flat[i].bool()]
            if bg_tokens_i.shape[0] == 0:
                hard_background_token_sets.append(
                    torch.zeros(actual_k, dim_semantic, device=fpn_flat.device)
                )
                continue

            bg_tokens_512 = self.visual_projection(bg_tokens_i)
            bg_normalized = F.normalize(bg_tokens_512, p=2, dim=-1)
            semantic_normalized = F.normalize(semantic_anchor[i:i + 1], p=2, dim=-1)
            similarity = torch.mm(bg_normalized, semantic_normalized.t()).squeeze(-1)

            k_i = min(actual_k, bg_tokens_i.shape[0])
            _, topk_idx = torch.topk(similarity, k=k_i, dim=0)
            neg_tokens_i = bg_tokens_512[topk_idx]
            if k_i < actual_k:
                pad = neg_tokens_i[-1:].expand(actual_k - k_i, -1)
                neg_tokens_i = torch.cat([neg_tokens_i, pad], dim=0)
            hard_background_token_sets.append(neg_tokens_i)

        return torch.stack(hard_background_token_sets, dim=0)
