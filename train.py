import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from model.primed import PRIMED
from model.checkpoint_utils import remap_legacy_primed_state_dict
from utility import AverageMeter, Eval_Fmeasure, log_parameter_count, mask_iou
from utils.avs_dataset_m3 import MS3Dataset_SAM
from utils.avs_dataset_s4 import S4Dataset_SAM
from utils.avs_dataset_refavs import RefAVSDataset_SAM

warnings.simplefilter("ignore", UserWarning)
criterionBCE = torch.nn.BCEWithLogitsLoss()

FIXED_AVS_TEXT = "the sounding object"
AVS_DATASETS = {"s4", "ms3"}


def resolve_task_text(dataset, text_captions, batch_size):
    """Use a task-level generic prompt for AVS and sample text for Ref-AVS."""
    if dataset in AVS_DATASETS:
        return [FIXED_AVS_TEXT] * batch_size
    return text_captions


def unpack_train_val_batch(batch, dataset):
    """Normalize task-specific dataset tuples for training and validation."""
    if dataset == "refavs":
        sam_imgs, clap_embeddings, text, masks, original_sizes, _, _, modality_prior_targets = batch
    else:
        sam_imgs, clap_embeddings, text, masks, original_sizes, _, modality_prior_targets = batch
    return sam_imgs, clap_embeddings, text, masks, original_sizes, modality_prior_targets

# Setup logging
def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

# def _iou_loss(pred, target):
#     pred = torch.sigmoid(pred)
#     inter = (pred * target).sum(dim=(1, 2))
#     union = (pred + target).sum(dim=(1, 2)) - inter
#     # Add epsilon to prevent division by zero
#     iou = 1 - (inter / (union + 1e-6))
#     return iou.mean()

def dice_loss(pred_mask, five_gt_masks):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, h, w)
        five_gt_masks (Tensor): (bs, h, w)
    """
    assert len(pred_mask.shape) == 3
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def compute_loss(pred_masks, masks_tensor, model=None, orthogonality_weight=0.01,
                modality_prior_kl_loss=None, kl_weight=0.1, sasa_loss=None, sasa_loss_weight=0.1):
    """Compute loss in FP32 to avoid NaN in BCEWithLogitsLoss
    
    Args:
        pred_masks: predictions (will be converted to FP32)
        masks_tensor: ground truth masks (will be converted to FP32)
        model: model instance for computing orthogonality loss (optional)
        orthogonality_weight: weight for orthogonality loss (default: 0.01)
        modality_prior_kl_loss: KL divergence loss from modality-prior decoding (optional)
        kl_weight: weight for KL divergence loss (default: 0.1)
        sasa_loss: SASA loss (optional)
        sasa_loss_weight: weight for SASA loss (default: 0.1)
    
    Returns:
        total_loss: combined loss
        loss_dict: dictionary with individual loss components
    """
    # Ensure FP32 computation
    pred_masks = pred_masks.float()
    masks_tensor = masks_tensor.float()
    
    # Segmentation losses
    bce_loss = criterionBCE(pred_masks, masks_tensor)
    iou_loss = dice_loss(pred_masks, masks_tensor)
    seg_loss = bce_loss + iou_loss
    
    total_loss = seg_loss
    loss_dict = {'seg_loss': seg_loss.item()}
    
    # Orthogonality regularization on the compressed tokens
    if model is not None:
        orthogonality_loss = model.compute_orthogonality_loss()
        total_loss = total_loss + orthogonality_weight * orthogonality_loss
        loss_dict['orthogonality_loss'] = orthogonality_loss.item()
    
    # Modality-prior KL loss (if provided)
    if modality_prior_kl_loss is not None:
        total_loss = total_loss + kl_weight * modality_prior_kl_loss
        loss_dict['kl_loss'] = modality_prior_kl_loss.item()
    
    # SASA loss (if provided)
    if sasa_loss is not None:
        total_loss = total_loss + sasa_loss_weight * sasa_loss
        loss_dict['sasa_loss'] = sasa_loss.item()
    
    loss_dict['total_loss'] = total_loss.item()
    
    return total_loss, loss_dict


def move_to_device(*tensors, device):
    """Move tensors to device."""
    return [tensor.to(device) for tensor in tensors]


def amp_enabled_for_device(device, requested):
    return requested and torch.device(device).type == "cuda"


def resolve_device(device):
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device '{device}', but CUDA is not available.")
        device_index = 0 if resolved.index is None else resolved.index
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device '{device}', but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(device_index)
    return str(resolved)


def train(train_loader, model, optimizer: AdamW, args, epoch, scaler=None, writer=None, scheduler=None):
    model.train()
    device = args.device
    total_loss = 0
    avg_meter_miou = AverageMeter("miou")
    avg_meter_F = AverageMeter("F_score")
    
    # Accumulators for epoch-level loss components
    total_seg_loss = 0
    total_kl_loss = 0
    total_sasa_loss = 0
    num_batches = 0
    
    # Calculate global iteration for iter-level logging
    # This ensures each iteration has a unique step number across all epochs
    iters_per_epoch = len(train_loader)
    
    # Create progress bar with disable auto-refresh
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120, mininterval=1.0)
    
    for iter_idx, batch in enumerate(pbar):
        # Unpack batch - text_captions is now raw string (processed in model)
        (
            sam_imgs_tensor,
            clap_embeddings,
            text_captions,
            masks_tensor,
            original_size_list,
            modality_prior_targets,
        ) = unpack_train_val_batch(batch, args.dataset)

        # Move tensors to the device (text_captions is List[str], not tensor)
        sam_imgs_tensor, clap_embeddings, masks_tensor, modality_prior_targets = move_to_device(
            sam_imgs_tensor, clap_embeddings, masks_tensor, modality_prior_targets, device=device
        )

        B_i, T_i, C_i, H_i, W_i = sam_imgs_tensor.shape
        B, T, C, H, W = masks_tensor.shape
        text_captions = resolve_task_text(args.dataset, text_captions, B)
        masks_tensor = masks_tensor.view(B * T, H, W)

        optimizer.zero_grad()

        # Use mixed precision training if enabled
        if amp_enabled_for_device(device, args.use_amp) and scaler is not None:
            # Forward pass in FP16
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
                pred_masks, modality_prior_kl_loss, sasa_loss = model(
                    sam_imgs_tensor, clap_embeddings, text_captions, original_size_list, modality_prior_targets=modality_prior_targets, gt_masks=masks_tensor
                )
            
            pred_masks = pred_masks.view(B * T, H, W)

            # Compute loss with orthogonal regularization and SASA loss
            loss, loss_dict = compute_loss(pred_masks, masks_tensor, model=model, orthogonality_weight=args.orthogonality_weight, 
                                            modality_prior_kl_loss=modality_prior_kl_loss, kl_weight=args.kl_weight,
                                            sasa_loss=sasa_loss, sasa_loss_weight=args.sasa_loss_weight)
            
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at iteration {iter_idx}, skipping")
                optimizer.zero_grad()
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale gradients and apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check for inf/nan gradients
            if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                   for p in model.parameters() if p.grad is not None):
                print(f"Warning: NaN/Inf gradients detected at iteration {iter_idx}, skipping")
                optimizer.zero_grad()
                continue
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            # Accumulate loss components for epoch-level statistics
            num_batches += 1
            if 'seg_loss' in loss_dict:
                total_seg_loss += loss_dict['seg_loss']
            if 'kl_loss' in loss_dict:
                total_kl_loss += loss_dict['kl_loss']
            if 'sasa_loss' in loss_dict:
                total_sasa_loss += loss_dict['sasa_loss']
        else:
            # Standard FP32 training
            pred_masks, modality_prior_kl_loss, sasa_loss = model(
                sam_imgs_tensor, clap_embeddings, text_captions, original_size_list, modality_prior_targets=modality_prior_targets, gt_masks=masks_tensor
            )
            pred_masks = pred_masks.view(B * T, H, W)
            
            # Compute loss with orthogonal regularization and SASA loss
            loss, loss_dict = compute_loss(pred_masks, masks_tensor, model=model, orthogonality_weight=args.orthogonality_weight, 
                                            modality_prior_kl_loss=modality_prior_kl_loss, kl_weight=args.kl_weight,
                                            sasa_loss=sasa_loss, sasa_loss_weight=args.sasa_loss_weight)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Accumulate loss components for epoch-level statistics
            num_batches += 1
            if 'seg_loss' in loss_dict:
                total_seg_loss += loss_dict['seg_loss']
            if 'kl_loss' in loss_dict:
                total_kl_loss += loss_dict['kl_loss']
            if 'sasa_loss' in loss_dict:
                total_sasa_loss += loss_dict['sasa_loss']
        
        # Step scheduler after optimizer update (for iter-based warmup)
        if scheduler is not None:
            scheduler.step()

        miou = mask_iou(pred_masks, masks_tensor)
        avg_meter_miou.add({"miou": miou})
        F_score = Eval_Fmeasure(pred_masks, masks_tensor)
        avg_meter_F.add({"F_score": F_score})
        
        # Log iteration-level metrics to TensorBoard
        if writer is not None:
            global_iter = (epoch - 1) * iters_per_epoch + iter_idx
            writer.add_scalar('Train_Iter/Loss', loss.item(), global_iter)
            writer.add_scalar('Train_Iter/mIoU', miou, global_iter)
            writer.add_scalar('Train_Iter/F_score', F_score, global_iter)
            # Log learning rate at iteration level to visualize warmup
            writer.add_scalar('Train_Iter/Learning_Rate', optimizer.param_groups[0]['lr'], global_iter)
            # Log individual loss components if available
            if 'seg_loss' in loss_dict:
                writer.add_scalar('Train_Iter/Seg_Loss', loss_dict['seg_loss'], global_iter)
            if 'kl_loss' in loss_dict:
                writer.add_scalar('Train_Iter/KL_Loss', loss_dict['kl_loss'], global_iter)
            if 'sasa_loss' in loss_dict:
                writer.add_scalar('Train_Iter/SASA_Loss', loss_dict['sasa_loss'], global_iter)
        
        # Update progress bar with current metrics
        current_avg_loss = total_loss / (iter_idx + 1)
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'miou': f'{miou:.4f}',
            'F': f'{F_score:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Train Epoch: Loss: {avg_loss:.4f}")
    train_metrics = {
        "miou": avg_meter_miou.pop("miou").item(),
        "F_score": avg_meter_F.pop("F_score"),
        "train_loss": avg_loss,
    }
    
    # Log epoch-level metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/mIoU', train_metrics["miou"], epoch)
        writer.add_scalar('Train/F_score', train_metrics["F_score"], epoch)
        
        # Log epoch-level loss components
        if num_batches > 0:
            if total_seg_loss > 0:
                writer.add_scalar('Train/Seg_Loss', total_seg_loss / num_batches, epoch)
            if total_kl_loss > 0:
                writer.add_scalar('Train/KL_Loss', total_kl_loss / num_batches, epoch)
            if total_sasa_loss > 0:
                writer.add_scalar('Train/SASA_Loss', total_sasa_loss / num_batches, epoch)
    
    return train_metrics


@torch.no_grad()
def validate(val_loader, model, args, scaler=None, writer=None, epoch=0):
    model.eval()
    avg_meter_miou = AverageMeter("miou")
    avg_meter_F = AverageMeter("F_score")
    device = args.device
    total_loss = 0

    # Create progress bar for validation
    pbar = tqdm(val_loader, desc="Validating", ncols=120, leave=True, position=0)
    
    for iter_idx, batch in enumerate(pbar):
        # Unpack batch - text_captions is now raw string (processed in model)
        (
            sam_imgs_tensor,
            clap_embeddings,
            text_captions,
            masks_tensor,
            original_size_list,
            modality_prior_targets,
        ) = unpack_train_val_batch(batch, args.dataset)

        # Move tensors to the device (text_captions is List[str], not tensor)
        sam_imgs_tensor, clap_embeddings, masks_tensor, modality_prior_targets = move_to_device(
            sam_imgs_tensor, clap_embeddings, masks_tensor, modality_prior_targets, device=device
        )

        B, T, C, H, W = masks_tensor.shape
        text_captions = resolve_task_text(args.dataset, text_captions, B)

        masks_tensor = masks_tensor.view(B * T, H, W)
        # Infer with mixed precision if enabled
        if amp_enabled_for_device(device, args.use_amp):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
                pred_masks, _, _ = model(
                    sam_imgs_tensor, clap_embeddings, text_captions, original_size_list, modality_prior_targets=modality_prior_targets
                )
        else:
            pred_masks, _, _ = model(
                sam_imgs_tensor, clap_embeddings, text_captions, original_size_list, modality_prior_targets=modality_prior_targets
            )

        pred_masks = pred_masks.view(B * T, H, W)

        # Compute metrics
        miou = mask_iou(pred_masks, masks_tensor)
        avg_meter_miou.add({"miou": miou})
        F_score = Eval_Fmeasure(pred_masks, masks_tensor)
        avg_meter_F.add({"F_score": F_score})

        # Compute loss for logging (masks_tensor and pred_masks already viewed at lines 306 and 318)
        val_loss, _ = compute_loss(pred_masks, masks_tensor)
        total_loss += val_loss.item()
        
        # Update progress bar
        current_avg_loss = total_loss / (iter_idx + 1)
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'miou': f'{miou:.4f}',
            'F': f'{F_score:.4f}'
        })

    avg_loss = total_loss / len(val_loader)
    # logging.info(f"Validation: Loss: {avg_loss:.4f}")
    # wandb.log({"val_loss": avg_loss})

    eval_metrics = {
        "miou": avg_meter_miou.pop("miou").item(),
        "F_score": avg_meter_F.pop("F_score"),
        "val_loss": avg_loss,
    }
    logging.info(f"Validation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Validation/Loss', avg_loss, epoch)
        writer.add_scalar('Validation/mIoU', eval_metrics["miou"], epoch)
        writer.add_scalar('Validation/F_score', eval_metrics["F_score"], epoch)
    
    return eval_metrics


@torch.no_grad()
def evaluate(test_loader, model, args):
    model.eval()
    avg_meter_miou = AverageMeter("miou")
    avg_meter_F = AverageMeter("F_score")
    device = args.device
    
    # Create progress bar for testing
    pbar = tqdm(test_loader, desc="Testing", ncols=120, leave=True, position=0)
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch - text_captions is now raw string (processed in model)
        # Test dataset only returns 7 values (no modality_prior_targets)
        (
            sam_imgs_tensor,
            clap_embeddings,
            text_captions,  # Raw text strings, model handles tokenization + encoding
            masks_tensor,
            original_size_list,
            _,
            _
        ) = batch

        # Move tensors to the device (text_captions is List[str], not tensor)
        sam_imgs_tensor, clap_embeddings, masks_tensor = move_to_device(
            sam_imgs_tensor, clap_embeddings, masks_tensor, device=device
        )

        B, T, C, H, W = masks_tensor.shape
        text_captions = resolve_task_text(args.dataset, text_captions, B)
        masks_tensor = masks_tensor.view(B * T, H, W)
        # Infer with mixed precision if enabled
        # Test mode: no modality_prior_targets (model will use default)
        if amp_enabled_for_device(device, args.use_amp):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
                pred_masks, _, _ = model(
                    sam_imgs_tensor, clap_embeddings, text_captions, original_size_list
                )
        else:
            pred_masks, _, _ = model(
                sam_imgs_tensor, clap_embeddings, text_captions, original_size_list
            )

        pred_masks = pred_masks.view(B * T, H, W)

        # Compute metrics
        miou = mask_iou(pred_masks, masks_tensor)
        avg_meter_miou.add({"miou": miou})
        F_score = Eval_Fmeasure(pred_masks, masks_tensor)
        avg_meter_F.add({"F_score": F_score})
        
        # Update progress bar
        pbar.set_postfix({
            'miou': f'{miou:.4f}',
            'F': f'{F_score:.4f}'
        })

    eval_metrics = {"miou": avg_meter_miou.pop("miou").item(), "F_score": avg_meter_F.pop("F_score")}
    logging.info(f"Test Evaluation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    return eval_metrics


def make_dataloader(args):
    assert args.dataset in ["ms3", "s4", "refavs"], "Dataset not supported"
    device = args.device  # Get device from args
    
    if args.dataset == "ms3":
        train_loader = DataLoader(
            MS3Dataset_SAM(split="train", device=device),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            MS3Dataset_SAM(split="val", device=device),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            MS3Dataset_SAM(split="test", device=device),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataset == "s4":
        train_loader = DataLoader(
            S4Dataset_SAM(split="train", device=device),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            S4Dataset_SAM(split="val", device=device),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            S4Dataset_SAM(split="test", device=device),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataset == "refavs":
        train_loader = DataLoader(
            RefAVSDataset_SAM(split="train", device=device),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            RefAVSDataset_SAM(split="val", device=device),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            RefAVSDataset_SAM(split="test", device=device),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def get_model(args):
    # get model and move to device
    model = PRIMED(args).to(device=args.device)
    return model


def main(args):
    torch.autograd.set_detect_anomaly(True)
    # Set up logging
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    setup_logger(os.path.join(args.save_path, "training_log.txt"))
    logging.info("Starting training process")
    
    # Log mixed precision status
    amp_enabled = amp_enabled_for_device(args.device, args.use_amp)
    if amp_enabled:
        logging.info("Mixed Precision Training (AMP) ENABLED")
    elif args.use_amp:
        logging.info(f"Mixed Precision Training requested but disabled for device: {args.device}")
    else:
        logging.info("Mixed Precision Training DISABLED - Using FP32")
    logging.info(f"Args:\n{args}")
    max_val_v = 1e-8
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if args.tensorboard:
        tensorboard_dir = os.path.join(args.save_path, 'tensorboard_logs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logging.info(f"TensorBoard logging enabled at: {tensorboard_dir}")

    # contain _adapter if using adapter
    name = f"{args.model_name}_{args.dataset}"
    if amp_enabled:
        name += "_amp"

    if args.wandb:
        if args.dataset == "s4":
            project_name = "avsbench_s4"
        else:
            project_name = "avsbench"

        wandb.init(
            project=project_name,
            config=args,
            name=name,
        )

    model = get_model(args)
    log_parameter_count(model, prefix="Training Model")

    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(remap_legacy_primed_state_dict(torch.load(args.resume, map_location=args.device)))
        logging.info("Model weights loaded successfully")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler with iter-based warmup
    # We'll use a custom step-based scheduler that updates per iteration
    # Note: scheduler will be initialized after train_loader is available
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if amp_enabled else None

    # Load data loaders
    train_loader, val_loader, test_loader = make_dataloader(args)
    val_interval = 1
    
    # Initialize iter-based scheduler after train_loader is available
    total_iters = len(train_loader) * args.max_epochs
    warmup_iters = max(1, int(args.warmup * total_iters))
    
    # Warmup phase: gradually increase lr from 0.1x to 1.0x over warmup_iters iterations
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    # Main phase: cosine annealing for remaining iterations
    main_scheduler = CosineAnnealingLR(optimizer, total_iters - warmup_iters, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup_scheduler, main_scheduler], 
                            milestones=[warmup_iters])
    logging.info(f"Using iter-based warmup scheduler: {warmup_iters} iterations warmup ({warmup_iters/total_iters:.2%} of {total_iters}) + cosine annealing")

    print(f"Training {args.model_name} on {args.dataset} for {args.max_epochs} epochs")

    for epoch in range(args.max_epochs):
        logging.info(f"Epoch [{epoch + 1}/{args.max_epochs}]")
        # Train
        train_metrics = train(train_loader, model, optimizer, args, epoch + 1, scaler=scaler, writer=writer, scheduler=scheduler)

        if args.wandb:
            wandb.log(
                {
                    "train_loss": train_metrics["train_loss"],
                    "train_miou": train_metrics["miou"],
                    "train_F_score": train_metrics["F_score"],
                }
            )
        
        # Log learning rate to TensorBoard
        if writer is not None:
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        # Note: scheduler.step() is now called per iteration in train() function, not here

        # Validate
        if (epoch + 1) % val_interval == 0:
            eval_result = validate(val_loader, model, args, scaler=scaler, writer=writer, epoch=epoch + 1)
            if args.wandb:
                wandb.log(
                    {
                        "val_miou": eval_result["miou"],
                        "val_F_score": eval_result["F_score"],
                        "val_loss": eval_result["val_loss"],
                    }
                )

            if eval_result["miou"] > max_val_v:
                max_val_v = eval_result["miou"]
                model_save_path = os.path.join(args.save_path, name + "_best.pth")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best model to {model_save_path}")

            # Always save latest checkpoint for resume
            latest_path = os.path.join(args.save_path, name + "_latest.pth")
            torch.save(model.state_dict(), latest_path)
            logging.info(f"Saved latest checkpoint to {latest_path}")

    model_save_path = os.path.join(args.save_path, name + "_best.pth")
    model.load_state_dict(remap_legacy_primed_state_dict(torch.load(model_save_path)))
    test_result = evaluate(test_loader, model, args)
    if args.wandb:
        wandb.log({"test_miou": test_result["miou"], "test_F_score": test_result["F_score"]})
    
    # Log test results to TensorBoard
    if writer is not None:
        writer.add_scalar('Test/mIoU', test_result["miou"], 0)
        writer.add_scalar('Test/F_score', test_result["F_score"], 0)
        writer.close()
        logging.info("TensorBoard writer closed")

if __name__ == "__main__":

    # Fix seed
    FixSeed = 42
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FixSeed)

    # Set deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="refavs", help="which subset of avsbench dataset")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--min_lr", type=float, default=5e-7)

    # Model
    parser.add_argument("--save_path", type=str, default="./weights")
    parser.add_argument("--model_name", type=str, default="primed")
    parser.add_argument("--output_mask_size", type=int, default=1024, help="Size of the output mask")
    
    # SAM2 model arguments
    parser.add_argument("--sam_scale", type=str, default="large", choices=["tiny", "small", "base", "large"], 
                        help="SAM2 model scale")
    parser.add_argument("--semantic_flow_beta", type=float, default=1.0, help="Weight for semantic flow loss")
    parser.add_argument("--sam2_checkpoint", type=str, default="./pretrained/sam2_hiera_base_plus.pt")
    parser.add_argument("--train_mask_decoder", action="store_true", help="Train SAM2 mask decoder")
    parser.add_argument("--train_prompt_encoder", action="store_true", help="Train SAM2 prompt encoder")
    parser.add_argument("--prompt_injection_mode", type=str, default="injection",
                        choices=["injection", "prompt_encoder"],
                        help="How dense prompt features are injected: 'injection' (via TwoWayTransformer additive) or 'prompt_encoder' (via SAM2 prompt encoder mask input)")
    parser.add_argument("--train_text_encoder", action="store_true", help="Train text encoder (DistilRoBERTa)")
    
    # Mixed precision training
    parser.add_argument("--use_amp", action="store_true", 
                        help="Enable Automatic Mixed Precision (AMP) training to reduce GPU memory usage")
    
    # Loss weights
    parser.add_argument("--orthogonality_weight", type=float, default=0.2,
                        help="Weight for orthogonality regularization on compressed tokens")
    parser.add_argument("--kl_weight", type=float, default=0.2,
                        help="Weight for modality-prior KL divergence loss")
    parser.add_argument("--sasa_loss_weight", type=float, default=1.0,
                        help="Weight for SASA loss")

    # Model components
    parser.add_argument("--num_compressed_tokens", type=int, default=4,
                        help="Number of compressed visual tokens shared across CBCF stages (default: 4)")

    # training
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard for logging")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pth) to resume training from")

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    
    # AVS subsets freeze the audio cached-memory update; Ref-AVS keeps it enabled.
    args.disable_cached_memory = args.dataset in AVS_DATASETS
    if args.disable_cached_memory:
        print(
            f"[AVS:{args.dataset}] fixed_text='{FIXED_AVS_TEXT}', "
            "disable_cached_memory=True"
        )
    
    main(args)
