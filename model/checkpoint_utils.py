import logging
import torch


def remap_legacy_primed_state_dict(state_dict):
    """Map draft PRIMED parameter names to the paper-aligned module names."""
    replacements = (
        ("multi_stage_fusion.stage_image_distillers.0.", "token_compressor_module.compressor."),
        ("multi_stage_fusion.compressors.0.", "token_compressor_module.compressor."),
        ("modality_classifier.classifier.", "modality_prior_decoder.decoder."),
        ("contrastive_loss_fn.fpn_to_semantic_proj.", "sasa_loss_fn.visual_projection."),
        ("semantic_prior_adapters.", "modality_prior_projectors."),
        ("cross_attn_group1_A.", "cbcf_round1_visual_attention."),
        ("cross_attn_group1_B.", "cbcf_round1_semantic_attention."),
        ("cross_attn_group2_A.", "cbcf_round2_visual_attention."),
        ("cross_attn_group2_B_shared.", "cbcf_round2_semantic_attention."),
        ("semantic_flow_gen.", "semantic_flow_generator."),
        ("vis_injection_projs.", "visual_injection_projections."),
        ("semantic_injection_projs.", "semantic_injection_projections."),
        ("prior_gamma", "modality_prior_scale"),
        (".cross_attn.", ".compression_attention."),
        (".audio_to_clip_proj.", ".audio_proj."),
        (".logit_scale", ".similarity_scale"),
    )

    remapped_state_dict = {}
    for key, value in state_dict.items():
        remapped_key = key
        for draft_name, official_name in replacements:
            remapped_key = remapped_key.replace(draft_name, official_name)
        remapped_state_dict[remapped_key] = value
    return remapped_state_dict

def load_sam2_checkpoint_compatible(model, ckpt_path, strict=False, verbose=True):
    if ckpt_path is None:
        if verbose:
            logging.info("No checkpoint provided, using random initialization")
        return [], []

    if verbose:
        logging.info(f"Loading SAM2 checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "memory_encoder.fuser.layers" in key and key.endswith(".gamma"):
            new_key = key.replace(".gamma", ".weight")
            logging.debug(f"Mapping key: {key} -> {new_key}")
        if new_key.startswith("module."):
            new_key = new_key[7:]
        new_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

    filtered_missing = [key for key in missing_keys if not key.endswith(".gamma")]
    filtered_unexpected = []
    for key in unexpected_keys:
        if key.endswith(".gamma"):
            weight_key = key.replace(".gamma", ".weight")
            if weight_key not in missing_keys:
                continue
        filtered_unexpected.append(key)

    if filtered_missing:
        logging.warning(f"Missing keys ({len(filtered_missing)}): {filtered_missing[:5]}...")
    if filtered_unexpected:
        logging.warning(f"Unexpected keys ({len(filtered_unexpected)}): {filtered_unexpected[:5]}...")

    if verbose:
        logging.info(
            f"SAM2 checkpoint loaded (missing: {len(filtered_missing)}, "
            f"unexpected: {len(filtered_unexpected)})"
        )

    return filtered_missing, filtered_unexpected
