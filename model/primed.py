import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

from model.modality_prior_decoder import ModalityPriorDecoder
from model.segment_anything_2.sam2.build_sam import build_sam2
from model.checkpoint_utils import load_sam2_checkpoint_compatible
from model.multi_stage_fusion import SharedTokenCompressor
from model.semantic_flow_generator import SemanticFlowGenerator
from model.semantic_prior_adapter import ModalityPriorProjector


class PRIMED(nn.Module):
    """
    PRIMED: PRior-Informed Modulation and Embedding Distillation for Ref-AVS.

    """
    def __init__(self, args):
        super(PRIMED, self).__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        
        # Get device from args
        self.device = getattr(args, 'device', 'cuda:0')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Output mask size configuration
        # If None, keep original SAM2 decoder output size (256x256)
        # Otherwise, resize to specified size
        self.output_mask_size = args.output_mask_size
        
        # Initialize SAM2
        self.sam2_model = self.init_sam2()
        
        # Initialize text encoder (DistilRoBERTa)
        self.text_encoder = self.init_text_encoder()
        
        # Initialize text tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "/nvme0/home/hyc/work/task1/PRIMED/pretrained/distilroberta-base",
            trust_remote_code=True
        )
        
        self.common_dim = 512
        self.visual_dim = 256
        self.audio_embedding_dim = 1024
        self.text_embedding_dim = 768
                
        # Shared Token Compressor

        stage4_trunk_dim = self.get_sam2_stage4_trunk_dim(args.sam_scale)
        self.token_compressor_module = SharedTokenCompressor(
            hidden_dim=self.common_dim,
            num_heads=8,
            image_feature_dims=[stage4_trunk_dim],
            num_compressed_tokens=getattr(args, 'num_compressed_tokens', 8),
        )

        self.sam_prompt_dim = 256
        
        self.semantic_tokens_proj = nn.Linear(self.common_dim, self.sam_prompt_dim)
        
        self.semantic_flow_generator = SemanticFlowGenerator(
            text_dim=self.text_embedding_dim,   # 768
            audio_dim=self.audio_embedding_dim,  # 1024
            common_dim=self.common_dim,          # 512
            num_heads=2,
            beta=getattr(args, 'semantic_flow_beta', 0.5),
            disable_cached_memory=args.disable_cached_memory,
        )
        
        trunk = self.sam2_model.image_encoder.trunk
        stage_dims = [trunk.blocks[se].dim_out for se in trunk.stage_ends]
        decoder_dim = self.sam_prompt_dim  # 256
        
        self.visual_injection_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stage_dims[i], decoder_dim, kernel_size=1, bias=True),
                nn.AdaptiveAvgPool2d((64, 64))
            ) for i in range(3)  # stages 0, 1, 2
        ])
        
        # Shape: [B*T, N, 512]
        self.semantic_injection_projections = nn.ModuleList([
            nn.Linear(self.common_dim, decoder_dim) for _ in range(3)
        ])

        self.modality_prior_decoder = ModalityPriorDecoder(
            input_dim=768,
            hidden_dim=256,
            num_classes=3  # audio, visual, audio-visual
        )
        
        from model.spatial_aware_semantic_alignment import SpatialAwareSemanticAlignmentLoss
        self.sasa_loss_fn = SpatialAwareSemanticAlignmentLoss(temperature=0.07)
        
        stage_dims = self._get_all_stage_trunk_dims(args.sam_scale)
        # The framework has three CBCF blocks on Hiera stages 1-3. Stage 4 is
        # reserved for the token compressor and the frozen FPN path.
        self.cbcf_stage_indices = tuple(range(len(stage_dims) - 1))
        cbcf_stage_dims = [stage_dims[i] for i in self.cbcf_stage_indices]
        semantic_dim = self.common_dim  # 512
        
        from model.cross_modal_biased_competition_fusion import MultiHeadCrossAttention
        
        self.cbcf_round1_visual_attention = nn.ModuleList([
            MultiHeadCrossAttention(
                query_dim=stage_dim,
                kv_dim=semantic_dim,
                num_heads=8
            ) for stage_dim in cbcf_stage_dims
        ])
        
        self.cbcf_round1_semantic_attention = nn.ModuleList([
            MultiHeadCrossAttention(
                query_dim=semantic_dim,
                kv_dim=stage_dim,
                num_heads=8
            ) for stage_dim in cbcf_stage_dims
        ])
        
        self.cbcf_round2_visual_attention = nn.ModuleList([
            MultiHeadCrossAttention(
                query_dim=stage_dim,
                kv_dim=semantic_dim,
                num_heads=8
            ) for stage_dim in cbcf_stage_dims
        ])
        
        self.cbcf_round2_semantic_attention = MultiHeadCrossAttention(
            query_dim=semantic_dim,
            kv_dim=semantic_dim,
            num_heads=8
        )
        
        self.stage_transition_convs = nn.ModuleList()
        for i in range(len(cbcf_stage_dims) - 1):

            conv = nn.Conv2d(
                in_channels=stage_dims[i],
                out_channels=stage_dims[i+1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True
            )
            self.stage_transition_convs.append(conv)
        
        # Modality prior: CLIP-like shared prior space, injected into the final CBCF stage.
        self.modality_prior_stage_index = self.cbcf_stage_indices[-1]
        self.modality_prior_projectors = nn.ModuleDict({
            str(self.modality_prior_stage_index): ModalityPriorProjector(
                visual_dim=stage_dims[self.modality_prior_stage_index],
                text_dim=self.text_embedding_dim,
                audio_dim=self.audio_embedding_dim,
                semantic_dim=self.common_dim,
            )
        })

        self.cls_token_proj = nn.Linear(512, 256)
        
        self.modality_prior_scale = torch.nn.Parameter(torch.tensor(0.0))
        
        # Feature size configuration
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def runtime_device(self):
        """Return the device the module currently lives on."""
        return next(self.parameters()).device
    
    def get_sam2_stage4_trunk_dim(self, sam_scale):
        """Return the final Hiera trunk channel dimension for the selected SAM2 scale."""

        trunk_dim_map = {
            "large": 1152,
            "base": 896,
            "small": 768,
            "tiny": 768,
        }
        if sam_scale not in trunk_dim_map:
            raise ValueError(f"Unknown SAM scale: {sam_scale}. Supported: {list(trunk_dim_map.keys())}")
        return trunk_dim_map[sam_scale]
    
    def _get_all_stage_trunk_dims(self, sam_scale):
        """Return trunk channel dimensions for all four Hiera stages."""

        base_dim_map = {
            "large": 144,
            "base": 112,
            "small": 96,
            "tiny": 96,
        }
        
        if sam_scale not in base_dim_map:
            raise ValueError(f"Unknown SAM scale: {sam_scale}")
        
        base_dim = base_dim_map[sam_scale]
        # Shape: [1, 2, 4, 8]
        stage_dims = [base_dim * (2**i) for i in range(4)]
        return stage_dims
    
    def init_sam2(self):
        """Initialize SAM2 model"""
        print(f"Initializing SAM2 with scale: {self.args.sam_scale}")
        
        if self.args.sam_scale == "large":
            config_file = "sam2_hiera_l.yaml"
        elif self.args.sam_scale == "tiny":
            config_file = "sam2_hiera_t.yaml"
        elif self.args.sam_scale == "base":
            config_file = "sam2_hiera_b+.yaml"
        elif self.args.sam_scale == "small":
            config_file = "sam2_hiera_s.yaml"
        else:
            raise ValueError(f"Unsupported SAM2 scale: {self.args.sam_scale}")
        
        model = build_sam2(config_file, ckpt_path=None, device=None)
        
        # Load checkpoint with compatibility handling
        ckpt_path = getattr(self.args, 'sam2_checkpoint', None)
        if ckpt_path:
            load_sam2_checkpoint_compatible(model, ckpt_path, strict=False)
        else:
            logging.info("No SAM2 checkpoint provided, using random initialization")
        
        # Freeze all parameters by default
        for param in model.parameters():
            param.requires_grad = False
        
        # Optionally train cross attention layers
        if getattr(self.args, 'train_sam2_cross_attn', False):
            print("Training SAM2 cross attention layers")
            for name, param in model.image_encoder.trunk.named_parameters():
                if "cross_attn" in name:
                    param.requires_grad = True
        
        # Optionally train mask decoder
        if getattr(self.args, 'train_mask_decoder', False):
            print("Training SAM2 mask decoder")
            model.sam_mask_decoder.train()
            for param in model.sam_mask_decoder.parameters():
                param.requires_grad = True
        
        # Optionally train prompt encoder
        if getattr(self.args, 'train_prompt_encoder', False):
            print("Training SAM2 prompt encoder")
            model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)
        
        model = model.to(self.device)
        return model
    
    def init_text_encoder(self):
        """Initialize text encoder (DistilRoBERTa)"""
        print("Initializing DistilRoBERTa text encoder")
        
        model = AutoModel.from_pretrained(
            "/nvme0/home/hyc/work/task1/PRIMED/pretrained/distilroberta-base",
            trust_remote_code=True
        )
        
        # Freeze text encoder parameters by default
        for param in model.parameters():
            param.requires_grad = False
        
        # Optionally train text encoder
        if getattr(self.args, 'train_text_encoder', False):
            print("Training text encoder")
            for param in model.parameters():
                param.requires_grad = True
        
        model = model.to(self.device)
        return model
    
    def preprocess_text_features(self, captions):
        """
        Tokenize and encode text captions with automatic padding.
        Following SAMWISE approach for handling variable-length text.
        
        Args:
            captions: List[str] or str - Text captions
        
        Returns:
            text_embeddings: torch.Tensor [B, max_N, 768] - Padded text token embeddings
            attention_mask: torch.Tensor [B, max_N] - Attention mask (1 for real tokens, 0 for padding)
        """
        # Ensure captions is a list
        if isinstance(captions, str):
            captions = [captions]
        
        # Tokenize with automatic padding
        batch_encoding = self.text_tokenizer(
            captions,
            add_special_tokens=True,
            padding=True,  # Automatically pad to longest sequence in batch
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        device = self.runtime_device()
        input_ids = batch_encoding['input_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)
        
        # Encode text
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Extract all tokens: [B, seq_len, 768]
            text_embeddings = outputs.last_hidden_state
        
        return text_embeddings, attention_mask
    
    def _forward_with_cbcf(self, images_sam_reshaped, semantic_flow,
                                         semantic_valid_mask=None,
                                         audio_embeddings=None, text_embeddings=None,
                                         modality_prior=None):
        """Implementation notes for this block."""
        trunk = self.sam2_model.image_encoder.trunk
        
        vis = trunk.patch_embed(images_sam_reshaped)
        if hasattr(trunk, '_get_pos_embed'):
            vis = vis + trunk._get_pos_embed(vis.shape[1:3])
        
        vis_outs = []
        
        stage_ends = trunk.stage_ends if hasattr(trunk, 'stage_ends') else []
        if not stage_ends:

            for block in trunk.blocks:
                vis = block(vis)
            vis_outs.append(vis.permute(0, 3, 1, 2))  # [B*T, H, W, C] -> [B*T, C, H, W]
            backbone_fpn, pos_enc = self.sam2_model.image_encoder.neck(vis_outs)
            return backbone_fpn, pos_enc, semantic_flow
        
        # semantic_flow: [B*T, N, 512]
        visual_branch = semantic_flow
        semantic_branch = semantic_flow
        
        hierarchical_visual_features = vis
        stage_trunk_features = []
        block_start = 0
        
        for stage_idx, stage_end in enumerate(stage_ends):
            for block_idx in range(block_start, stage_end + 1):
                hierarchical_visual_features = trunk.blocks[block_idx](hierarchical_visual_features)
            
            stage_trunk_features.append(hierarchical_visual_features.detach())  # [B*T, H, W, trunk_dim]
            
            if stage_idx == len(stage_ends) - 1:  # Stage 3 (final stage)
                B_T_s4, H_s4, W_s4, trunk_dim_s4 = hierarchical_visual_features.shape
                vis_flat_s4 = hierarchical_visual_features.reshape(B_T_s4, H_s4 * W_s4, trunk_dim_s4)
                compressed_visual_tokens = self.token_compressor_module.compressor(vis_flat_s4)
                # compressed_visual_tokens: [B*T, 8, 512]
            
            block_start = stage_end + 1
        
        cbcf_stages = self.cbcf_stage_indices
        vis_outs = []
        
        stage_visual_features = []
        stage_semantic_features = []
        
        for stage_idx in range(len(stage_ends)):

            vis = stage_trunk_features[stage_idx]
            
            if stage_idx in cbcf_stages:
                B_T, H, W, trunk_dim = vis.shape
                
                # Modality prior attention bias, injected at the final CBCF stage only.
                modality_prior_bias = None
                if stage_idx == self.modality_prior_stage_index:
                    modality_prior_bias = self._compute_modality_prior_bias(
                        vis, audio_embeddings, text_embeddings, modality_prior, stage_idx
                    )  # [B*T, 1, H*W]
                
                fused_visual_tokens, semantic_branch, fused_visual_features = self._apply_cbcf_stage(
                    visual_branch, semantic_branch, vis, compressed_visual_tokens, stage_idx,
                    modality_prior_bias=modality_prior_bias, semantic_valid_mask=semantic_valid_mask
                )
                
                stage_visual_features.append(fused_visual_features.permute(0, 3, 1, 2))  # [B*T, trunk_dim, H, W]
                stage_semantic_features.append(semantic_branch.clone())  # [B*T, N, 512]
                
                vis_outs.append(fused_visual_features.permute(0, 3, 1, 2))  # [B*T, C, H, W]
                
                if stage_idx < self.cbcf_stage_indices[-1]:
                    next_stage_idx = stage_idx + 1
                    next_H, next_W = stage_trunk_features[next_stage_idx].shape[1:3]
                    
                    propagated_visual_features = fused_visual_tokens  # [B*T, trunk_dim, H, W]
                    
                    propagated_visual_features = self.stage_transition_convs[stage_idx](
                        propagated_visual_features
                    )  # [B*T, next_trunk_dim, next_H_approx, next_W_approx]
                    
                    if propagated_visual_features.shape[2:4] != (next_H, next_W):
                        propagated_visual_features = torch.nn.functional.interpolate(
                            propagated_visual_features,
                            size=(next_H, next_W),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    visual_gate = torch.sigmoid(propagated_visual_features)
                    
                    next_vis = stage_trunk_features[next_stage_idx].permute(0, 3, 1, 2)  # [B*T, trunk_dim, next_H, next_W]
                    next_vis_modulated = next_vis + next_vis * visual_gate
                    
                    stage_trunk_features[next_stage_idx] = next_vis_modulated.permute(0, 2, 3, 1)
            else:

                vis_outs.append(vis.permute(0, 3, 1, 2))
        
        backbone_fpn, pos_enc = self.sam2_model.image_encoder.neck(vis_outs)
        
        scalp = self.sam2_model.image_encoder.scalp
        if scalp > 0:

            backbone_fpn = backbone_fpn[:-scalp]
            pos_enc = pos_enc[:-scalp]
        
        updated_semantic_flow = semantic_branch  # [B*T, N, 512]
        return backbone_fpn, pos_enc, updated_semantic_flow, stage_visual_features, stage_semantic_features
    
    def _apply_cbcf_stage(
        self, visual_branch, semantic_branch, vis, compressed_tokens, stage_idx,
        modality_prior_bias=None, semantic_valid_mask=None,
    ):
        """Fuse semantic flow and visual features for one Hiera stage."""
        B_T, H, W, trunk_dim = vis.shape
        N_flow = visual_branch.shape[1]
        
        # Shape: [B*T, H, W, trunk_dim], [B*T, H*W, trunk_dim]
        vis_flat = vis.reshape(B_T, H * W, trunk_dim)

        semantic_key_padding_mask = None
        semantic_token_weights = None
        if semantic_valid_mask is not None:
            if semantic_valid_mask.shape != semantic_branch.shape[:2]:
                raise ValueError(
                    "semantic_valid_mask must match semantic flow token dimensions, "
                    f"got {tuple(semantic_valid_mask.shape)} and {tuple(semantic_branch.shape[:2])}"
                )
            semantic_valid_mask = semantic_valid_mask.to(device=semantic_branch.device, dtype=torch.bool)
            semantic_key_padding_mask = ~semantic_valid_mask
            semantic_token_weights = semantic_valid_mask.unsqueeze(-1).to(semantic_branch.dtype)
        
        # modality_prior_bias: [B*T, 1, H*W]
        attn_mask_vis_q = None  # Shape: [B*T, 1, H*W, 1]
        attn_mask_vis_kv = None  # Shape: [B*T, 1, 1, H*W]
        attn_mask_global = None
        if modality_prior_bias is not None:
            attn_mask_vis_q = modality_prior_bias.unsqueeze(-1)    # [B*T, 1, H*W, 1]
            attn_mask_vis_kv = modality_prior_bias.unsqueeze(2)    # [B*T, 1, 1, H*W]
            attn_mask_global = modality_prior_bias.mean(dim=-1, keepdim=True).unsqueeze(2)  # [B*T, 1, 1, 1]
        
        visual_round1_delta = self.cbcf_round1_visual_attention[stage_idx](
            query=vis_flat,  # [B*T, H*W, trunk_dim]
            key_value=visual_branch,  # [B*T, N, 512]
            attn_mask=attn_mask_vis_q,  # [B*T, 1, H*W, 1] or None
            key_padding_mask=semantic_key_padding_mask,
        )  # [B*T, H*W, trunk_dim]
        visual_round1_features = visual_round1_delta + vis_flat
        
        semantic_round1_delta = self.cbcf_round1_semantic_attention[stage_idx](
            query=semantic_branch,  # [B*T, N, 512]
            key_value=vis_flat,  # [B*T, H*W, trunk_dim]
            attn_mask=attn_mask_vis_kv  # [B*T, 1, 1, H*W] or None
        )  # [B*T, N, 512]
        semantic_round1_features = semantic_round1_delta + semantic_branch
        
        visual_round2_delta = self.cbcf_round2_visual_attention[stage_idx](
            query=visual_round1_features,  # [B*T, H*W, trunk_dim]
            key_value=compressed_tokens,  # [B*T, 8, 512]
            attn_mask=attn_mask_vis_q  # [B*T, 1, H*W, 1] or None
        )  # [B*T, H*W, trunk_dim]
        fused_visual_tokens = visual_round2_delta + visual_round1_features
        
        semantic_round2_delta = self.cbcf_round2_semantic_attention(
            query=semantic_round1_features,  # [B*T, N, 512]
            key_value=compressed_tokens,  # [B*T, 8, 512]
            attn_mask=attn_mask_global
        )  # [B*T, N, 512]
        fused_semantic_tokens = semantic_round2_delta + semantic_round1_features

        if semantic_token_weights is not None:
            fused_semantic_tokens = fused_semantic_tokens * semantic_token_weights
        
        fused_visual_map = fused_visual_tokens.reshape(B_T, H, W, trunk_dim).permute(0, 3, 1, 2)  # [B*T, trunk_dim, H, W]
        
        fused_visual_features = fused_visual_tokens.reshape(B_T, H, W, trunk_dim)
        
        fused_semantic_flow = fused_semantic_tokens
        
        fused_visual_map = fused_visual_map  # [B*T, trunk_dim, H, W]
        
        return fused_visual_map, fused_semantic_flow, fused_visual_features
    
    def _compute_modality_prior_bias(self, vis, audio_embeddings, text_embeddings, modality_prior, stage_idx):
        """Compute a spatial modality-prior logit bias from visual, text, audio, and modality signals."""
        B_T, H, W, trunk_dim = vis.shape
        
        vis_flat = vis.reshape(B_T, H * W, trunk_dim)  # [B*T, H*W, trunk_dim]
        visual_sem, text_sem, audio_sem, _ = self.modality_prior_projectors[str(stage_idx)](
            visual_features=vis_flat,
            text_features=text_embeddings,
            audio_features=audio_embeddings,
        )
        normalized_visual = visual_sem
        normalized_audio = audio_sem
        normalized_text = text_sem
        
        eps = 1e-6
        
        audio_similarity_raw = torch.bmm(normalized_audio, normalized_visual.transpose(1, 2))  # Shape: [B*T, 1, H*W], [-1, 1]
        audio_similarity_positive = (audio_similarity_raw + 1.0) / 2.0  # Shape: [0, 1]
        audio_similarity = torch.clamp(audio_similarity_positive, eps, 1.0 - eps)
        
        visual_similarity_raw = torch.bmm(normalized_text, normalized_visual.transpose(1, 2))  # Shape: [B*T, 1, H*W], [-1, 1]
        visual_similarity_positive = (visual_similarity_raw + 1.0) / 2.0  # Shape: [0, 1]
        visual_similarity = torch.clamp(visual_similarity_positive, eps, 1.0 - eps)
        
        audio_prior_weight = modality_prior[:, 0].reshape(B_T, 1, 1)   # [B*T, 1, 1]
        visual_prior_weight = modality_prior[:, 1].reshape(B_T, 1, 1)   # [B*T, 1, 1]
        joint_prior_weight = modality_prior[:, 2].reshape(B_T, 1, 1)  # [B*T, 1, 1]
        
        modality_prior_bias = visual_prior_weight * visual_similarity + audio_prior_weight * audio_similarity + joint_prior_weight * (visual_similarity * audio_similarity)  # [B*T, 1, H*W]
        modality_prior_bias = torch.clamp(modality_prior_bias, eps, 1.0 - eps)

        modality_prior_bias = torch.log(modality_prior_bias) - torch.log(1.0 - modality_prior_bias)  # logit
        
        modality_prior_bias = self.modality_prior_scale * modality_prior_bias
        
        return modality_prior_bias
    
    def forward(self, images_sam, clap_embeddings, text_captions, original_size_lists, modality_prior_targets=None, gt_masks=None):
        """Run the forward pass."""
        B, T, C, H, W = images_sam.shape
        
        text_embeddings, text_attention_mask = self.preprocess_text_features(text_captions)

        global_text_feature = text_embeddings[:, 0, :]  # [B, 768] - CLS token
        modality_prior_logits = self.modality_prior_decoder(global_text_feature)  # [B, 3]
        modality_prior = torch.softmax(modality_prior_logits, dim=-1)  # [B, 3] - [audio_prior_weight, visual_prior_weight, joint_prior_weight]
        
        modality_prior_kl_loss = None
        if modality_prior_targets is not None and self.training:
            log_probs_pred = torch.nn.functional.log_softmax(modality_prior_logits, dim=-1)
            modality_prior_kl_loss = torch.nn.functional.kl_div(
                log_probs_pred, 
                modality_prior_targets, 
                reduction='batchmean'
            )
        
        # Shape: [B, T_audio, 1024], [B, L, 768]
        # Shape: [B, L+T_audio, 512]
        audio_embeddings = clap_embeddings  # [B, T_audio, 1024]
        semantic_flow = self.semantic_flow_generator(
            text_features=text_embeddings,  # Shape: [B, L, 768]
            audio_features=audio_embeddings,  # [B, T_audio, 1024]
            text_attention_mask=text_attention_mask,
        )  # [B, L+T_audio, 512]

        audio_valid_mask = torch.ones(
            B, audio_embeddings.shape[1], device=text_attention_mask.device, dtype=torch.bool
        )
        semantic_valid_mask = torch.cat(
            [text_attention_mask.to(dtype=torch.bool), audio_valid_mask], dim=1
        )  # [B, L+T_audio]
        
        # Shape: [B, T, L+T_audio, 512], [B*T, L+T_audio, 512]
        semantic_flow = semantic_flow.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, L+T_audio, 512]
        semantic_flow = semantic_flow.reshape(B * T, -1, self.common_dim)  # [B*T, L+T_audio, 512]
        semantic_valid_mask = semantic_valid_mask.unsqueeze(1).expand(-1, T, -1)
        semantic_valid_mask = semantic_valid_mask.reshape(B * T, -1)  # [B*T, L+T_audio]
                    
        images_sam_reshaped = images_sam.reshape(B * T, C, H, W)
        
        # Broadcast the per-clip prior signals over the T frames.
        expanded_audio_embeddings = audio_embeddings.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, self.audio_embedding_dim)  # [B*T, T_audio, 1024]
        expanded_text_embeddings = text_embeddings.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, self.text_embedding_dim)    # [B*T, max_N, 768]
        expanded_modality_prior = modality_prior.unsqueeze(1).expand(-1, T, -1).reshape(B * T, 3)  # [B*T, 3]
        
        backbone_fpn, pos_enc, cbcf_semantic_flow, stage_visual_features, stage_semantic_features = self._forward_with_cbcf(
            images_sam_reshaped=images_sam_reshaped,
            semantic_flow=semantic_flow,  # [B*T, L+T_audio, 512]
            semantic_valid_mask=semantic_valid_mask,
            audio_embeddings=expanded_audio_embeddings,
            text_embeddings=expanded_text_embeddings,
            modality_prior=expanded_modality_prior
        )
        
        sasa_loss = None
        sasa_loss_info = None
        if gt_masks is not None and self.training:
            # Shape: [-2]
            stage3_fpn = backbone_fpn[-2]  # [B*T, 256, H3, W3]
            
            # Shape: [B, T, H_gt, W_gt], [B*T, H_gt, W_gt]
            if gt_masks.dim() == 4 and gt_masks.shape[1] == 1:
                gt_masks_flat = gt_masks.squeeze(1).reshape(B * T, -1)  # [B*T, H_gt*W_gt]
            else:
                gt_masks_flat = gt_masks.reshape(B * T, -1)
            
            H_gt = gt_masks.shape[-2]
            W_gt = gt_masks.shape[-1]
            gt_masks_reshaped = gt_masks_flat.reshape(B * T, H_gt, W_gt)  # [B*T, H_gt, W_gt]
            
            sasa_loss, sasa_loss_info = self.sasa_loss_fn(
                fpn_features=stage3_fpn,
                gt_masks=gt_masks_reshaped,
                semantic_flow=semantic_flow,  # [B*T, L+T_audio, 512]
                top_k=10,
                semantic_valid_mask=semantic_valid_mask,
            )
            
        
        # Shape: [B*T, N, 512]
        semantic_no_cls = cbcf_semantic_flow[:, 1:, :]  # [B*T, N-1, 512]

        semantic_no_cls_mask = semantic_valid_mask[:, 1:].unsqueeze(1).to(semantic_no_cls.dtype)
        semantic_no_cls_t = semantic_no_cls.transpose(1, 2)  # [B*T, 512, N-1]
        semantic_tokens_4_sum = torch.nn.functional.adaptive_avg_pool1d(
            semantic_no_cls_t * semantic_no_cls_mask, 4
        )
        semantic_tokens_4_count = torch.nn.functional.adaptive_avg_pool1d(
            semantic_no_cls_mask, 4
        )
        semantic_tokens_4 = semantic_tokens_4_sum / semantic_tokens_4_count.clamp(min=1e-6)
        semantic_tokens_4 = semantic_tokens_4.transpose(1, 2)  # [B*T, 4, 512]

        semantic_tokens_4_proj = self.semantic_tokens_proj(semantic_tokens_4)  # [B*T, 4, 256]
        
        cls_token_fused = cbcf_semantic_flow[:, 0, :]  # [B*T, 512]
        cls_token_fused = self.cls_token_proj(cls_token_fused)  # [B*T, 256]
        cls_token_fused = cls_token_fused.unsqueeze(1)  # [B*T, 1, 256]
        coarse_mask_feature = backbone_fpn[0]  # [B*T, 256, H, W]
        
        sam_backbone_output = {
            "vision_features": backbone_fpn[-1],  # Shape: [B*T, 256, 64, 64]
            "vision_pos_enc": pos_enc,
            "backbone_fpn": backbone_fpn,
        }
        
        if self.sam2_model.use_high_res_features_in_sam:

            sam_backbone_output["backbone_fpn"][0] = self.sam2_model.sam_mask_decoder.conv_s0(
                sam_backbone_output["backbone_fpn"][0]  # Shape: [B*T, 256, 256, 256], [B*T, 32, 256, 256]
            )

            sam_backbone_output["backbone_fpn"][1] = self.sam2_model.sam_mask_decoder.conv_s1(
                sam_backbone_output["backbone_fpn"][1]  # Shape: [B*T, 256, 128, 128], [B*T, 64, 128, 128]
            )
    
        output_size = self.output_mask_size
        
        # prompt_encoder
        if self.args.prompt_injection_mode == 'prompt_encoder':
            coarse_masks = torch.einsum('bqc,bchw->bqhw', cls_token_fused, coarse_mask_feature)

            pred_masks = self.get_pred_masks(
                images_sam_reshaped,
                sam_backbone_output,
                semantic_tokens_4_proj,
                original_size_lists,
                coarse_masks,
                output_size=output_size,
                dense_injections=None,
                sparse_injections=None,
            )
        
        else:

            # Shape: [B*T, 4096, 256]
            dense_injections = []
            for i, vis_raw in enumerate(stage_visual_features):
                vis_proj = self.visual_injection_projections[i](vis_raw)  # [B*T, 256, 64, 64]
                vis_flat = vis_proj.flatten(2).permute(0, 2, 1)  # [B*T, 4096, 256]
                dense_injections.append(vis_flat)
            dense_injections = dense_injections[::-1]  # Shape: [stage2, stage1, stage0]
            
            # Shape: [B*T, 1, 256]
            sparse_injections = []
            for i, flow_b in enumerate(stage_semantic_features):
                sem_proj = self.semantic_injection_projections[i](flow_b)  # [B*T, N, 256]
                semantic_weights = semantic_valid_mask.unsqueeze(-1).to(sem_proj.dtype)
                sem_pooled = (sem_proj * semantic_weights).sum(dim=1, keepdim=True)
                sem_pooled = sem_pooled / semantic_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
                sparse_injections.append(sem_pooled)
            sparse_injections = sparse_injections[::-1]  # Shape: [stage2, stage1, stage0]
            
            # Get prediction masks with empty prompt encoder + stage 0-2 injections
            pred_masks = self.get_pred_masks(
                images_sam_reshaped,
                sam_backbone_output,
                None,
                original_size_lists,
                None,
                output_size=output_size,
                dense_injections=dense_injections,  # Shape: [stage2->Block0, stage1->Block1, stage0->FinalAttn]
                sparse_injections=sparse_injections,  # Shape: [stage2->Block0, stage1->Block1, stage0->FinalAttn]
            )
        
        return pred_masks, modality_prior_kl_loss, sasa_loss

    def get_pred_masks(
        self,
        images,
        sam_backbone_output,
        sparse_prompt_tokens,  # Shape: [B*T, N, 256]
        original_size_list,
        dense_prompt_features,
        output_size=None,
        dense_injections=None,
        sparse_injections=None,
    ):
        """
        Generate prediction masks using pre-computed backbone features.
        
        Args:
            images: [B*T, C, H, W] - Input images (used only for batch size)
            sam_backbone_output: dict - Pre-computed backbone features from _forward_with_cbcf
                                       Contains 'vision_features', 'vision_pos_enc', 'backbone_fpn'
            sparse_prompt_tokens: [B*T, N, 256] - Sparse prompt embeddings
            original_size_list: List of original image sizes [[h, w]]
            dense_prompt_features: [B*T, 1, H, W] - Dense prompt features (coarse masks)
            output_size: Optional tuple (H, W) or int for output mask size
            dense_injections: List of [B*T, 4096, 256] - Visual injection into keys
            sparse_injections: List of [B*T, 1, 256] - Semantic injection into queries
        
        Returns:
            pred_mask: [B*T, 1, H, W] - Predicted masks at target resolution
        """
        BT = images.shape[0]
        
        _, image_embeddings, _, feat_sizes = self.sam2_model._prepare_backbone_features(
            sam_backbone_output
        )
        image_embeddings = [_.to(images.dtype) for _ in image_embeddings]
        
        # Add no_mem_embed if needed
        if self.sam2_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.sam2_model.no_mem_embed
        
        # Reshape features to the format expected by mask decoder
        feats = [
            feat.permute(1, 2, 0).contiguous().view(BT, -1, *feat_size)
            for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        if self.args.prompt_injection_mode == 'prompt_encoder':
            # Prepare sparse embeddings (already projected to 256-dim)
            sparse_embeddings = sparse_prompt_tokens  # [BT, N, 256]
        
            # Encode dense prompt features through SAM prompt encoder
            if dense_prompt_features is not None:
                _, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=dense_prompt_features,
                    text_embeds=None,
                )
            else:
                _, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=None,
                )
        else:
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=None,
            )
            sparse_embeddings = sparse_embeddings.expand(BT, -1, -1)
            dense_embeddings = dense_embeddings.expand(BT, -1, -1, -1)
        
        # Align data types
        high_res_features = _features["high_res_feats"]
        sparse_embeddings = sparse_embeddings.to(dense_embeddings.dtype)
        
        # Decode masks using SAM2 mask decoder
        low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
            dense_injections=dense_injections,
            sparse_injections=sparse_injections,
        )
        
        # Select best mask based on IoU predictions
        sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

        # Postprocess masks to target size
        if output_size is not None:
            pred_mask = self.postprocess_masks(low_res_masks, orig_hw=output_size)
        else:
            orig_size = original_size_list[0]  # [h, w]
            pred_mask = self.postprocess_masks(low_res_masks, orig_hw=orig_size)

        return pred_mask

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """Postprocess masks to original size"""
        masks = masks.float()
        if isinstance(orig_hw, int):
            orig_hw = (orig_hw, orig_hw)
        masks = torch.nn.functional.interpolate(
            masks, orig_hw, mode="bilinear", align_corners=False
        )
        return masks
    
    def compute_orthogonality_loss(self):
        """Compute the Token Compressor orthogonality regularization."""

        return self.token_compressor_module.compute_orthogonality_loss()
