import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.primed import PRIMED
from model.checkpoint_utils import remap_legacy_primed_state_dict
from utility import AverageMeter, Eval_Fmeasure, log_parameter_count, mask_iou, metric_s_for_null
from utils.avs_dataset_m3 import MS3Dataset_SAM
from utils.avs_dataset_s4 import S4Dataset_SAM
from utils.avs_dataset_refavs import RefAVSDataset_SAM

warnings.simplefilter("ignore", UserWarning)

FIXED_AVS_TEXT = "the sounding object"


def get_model(args):
    # Always use the current modular PRIMED path.
    model = PRIMED(args).to(device=args.device)
    return model

# Setup logging
def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def move_to_device(*tensors, device):
    return [tensor.to(device) for tensor in tensors]

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

def unwrap_model_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output

@torch.no_grad()
def evaluate_one_second_avs(model, sam_imgs_tensor, clap_embeddings, original_size_list, args):
    """Run each frame as an independent 1-second AVS sample."""
    B, T = sam_imgs_tensor.shape[:2]
    if clap_embeddings.dim() != 3:
        raise ValueError(
            f"--one_second_avs_eval expects frame-level audio [B, T, C], got {tuple(clap_embeddings.shape)}"
        )
    if clap_embeddings.shape[1] < T:
        raise ValueError(
            f"--one_second_avs_eval needs at least {T} audio tokens, got {clap_embeddings.shape[1]}"
        )

    fixed_text = [FIXED_AVS_TEXT] * B
    pred_per_frame = []
    for frame_idx in range(T):
        frame_imgs = sam_imgs_tensor[:, frame_idx:frame_idx + 1]
        frame_audio = clap_embeddings[:, frame_idx:frame_idx + 1]
        frame_size = (
            [original_size_list[frame_idx]]
            if isinstance(original_size_list, (list, tuple)) and len(original_size_list) > frame_idx
            else original_size_list
        )
        pred_mask = unwrap_model_output(
            model(frame_imgs, frame_audio, fixed_text, frame_size)
        )
        pred_per_frame.append(pred_mask.view(B, 1, *pred_mask.shape[-2:]))

    return torch.cat(pred_per_frame, dim=1)

def save_mask(pred_masks, vid, fid, exp, args):
    save_base_path = args.save_path
    mask_save_path = os.path.join(save_base_path, f"pred_masks_{args.model_name}_{args.refavs_split}")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path, exist_ok=True)

    pred_masks = torch.permute(pred_masks, (1, 0, 2, 3))
    pred_masks = pred_masks.cpu().numpy().astype(np.uint8)
    pred_masks *= 255
    
    for idx in range(len(pred_masks)):
        one_mask = pred_masks[idx]

        one_mask = np.squeeze(one_mask, axis=0)

        # Save the mask as a PNG image
        output_name = f"{idx}.png"
        im = Image.fromarray(one_mask).convert('L')  # 'L' mode for grayscale image
        im_path = os.path.join(mask_save_path, vid, f"fid_{fid}", exp)
        if not os.path.exists(im_path):
            os.makedirs(im_path, exist_ok=True)
        im.save(os.path.join(im_path, output_name), format='PNG')

@torch.no_grad()
def evaluate(test_loader, model, args):
    model.eval()
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    null_s_list = []
    device = args.device
    save_path = args.save_path
    
    # Create progress bar
    pbar = tqdm(test_loader, desc="Evaluating", ncols=120, leave=True, position=0)
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch based on dataset type
        if args.dataset == "refavs":
            (
                sam_imgs_tensor, 
                clap_embeddings, 
                exp,
                masks_tensor, 
                original_size_list,
                vid,
                fid
            ) = batch
            
            (
                sam_imgs_tensor, 
                clap_embeddings, 
                masks_tensor
            ) = move_to_device(
                sam_imgs_tensor, clap_embeddings, masks_tensor, device=device
            )
            text_captions = exp
        else:
            (
                sam_imgs_tensor,
                clap_embeddings,
                _,
                masks_tensor,
                original_size_list,
                video_name,
                _
            ) = batch
            
            sam_imgs_tensor, clap_embeddings, masks_tensor = move_to_device(
                sam_imgs_tensor, clap_embeddings, masks_tensor, device=device
            )
            text_captions = [FIXED_AVS_TEXT] * sam_imgs_tensor.shape[0]

        B, T, C, H, W = masks_tensor.shape
        masks_tensor = masks_tensor.view(B*T, H, W)
        
        # Infer
        if args.one_second_avs_eval:
            pred_masks = evaluate_one_second_avs(
                model, sam_imgs_tensor, clap_embeddings, original_size_list, args
            )
        elif args.dataset == "refavs" and text_captions is not None:
            pred_masks = unwrap_model_output(
                model(sam_imgs_tensor, clap_embeddings, text_captions, original_size_list)
            )
        else:
            pred_masks = unwrap_model_output(
                model(sam_imgs_tensor, clap_embeddings, text_captions, original_size_list)
            )

        pred_masks = pred_masks.view(B*T, H, W)

        # Compute metrics
        if args.dataset == "refavs" and args.refavs_split == "test_n":
            null_s = metric_s_for_null(pred_masks)
            null_s_list.append(null_s.cpu().numpy())
            pbar.set_postfix({'S': f'{null_s:.4f}'})
        else:
            miou = mask_iou(pred_masks, masks_tensor)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(pred_masks, masks_tensor)
            avg_meter_F.add({'F_score': F_score})
            pbar.set_postfix({'miou': f'{miou:.4f}', 'F': f'{F_score:.4f}'})

        # Save masks
        if args.save:
            if args.dataset == "refavs":
                vid_to_save = vid[0]
                fid_to_save = fid[0].item()
                exp_to_save = exp[0]
                orig_h, orig_w = original_size_list[0]
                pred_masks_original = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                save_mask(pred_masks_original.view(B, T, orig_h, orig_w), vid_to_save, fid_to_save, exp_to_save, args)
            else:
                sample_name = video_name[0] if isinstance(video_name, (list, tuple)) else str(video_name)
                save_mask(pred_masks.view(B, T, H, W), sample_name, "all", FIXED_AVS_TEXT, args)
    
    # Log results
    if args.dataset == "refavs" and args.refavs_split == "test_n":
        miou = np.mean(np.array(null_s_list))
        eval_metrics = {'S': miou}
        logging.info(f"Test Evaluation: S: {eval_metrics['S']:.4f}")
    else:
        eval_metrics = {
            'miou': avg_meter_miou.pop('miou').item(),
            'F_score': avg_meter_F.pop('F_score')
        }
        logging.info(f"Test Evaluation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    return eval_metrics

def main(args):
    # Set up logging
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    setup_logger(os.path.join(args.save_path, f"test_log_{args.model_name}.txt"))
    logging.info("Starting evaluation process")
    
    model = get_model(args)
    log_parameter_count(model, prefix="Evaluation Model")
    if args.weight_path:
        print(f"Loading weights from {args.weight_path}")
        state_dict = torch.load(args.weight_path, map_location=args.device)
        state_dict = remap_legacy_primed_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded weights from {args.weight_path}")
    
    device = args.device
    if args.dataset == "ms3":
        test_loader = DataLoader(MS3Dataset_SAM(split='test', device=device), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    elif args.dataset == "s4":
        test_loader = DataLoader(S4Dataset_SAM(split='test', device=device), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    elif args.dataset == "refavs":
        if args.refavs_split == "test_s":
            test_loader = DataLoader(RefAVSDataset_SAM(split='test_s', device=device), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        elif args.refavs_split == "test_u":
            test_loader = DataLoader(RefAVSDataset_SAM(split='test_u', device=device), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        elif args.refavs_split == "test_n":
            test_loader = DataLoader(RefAVSDataset_SAM(split='test_n', device=device), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Evaluate
    eval_result = evaluate(test_loader, model, args)
    
    # Log final results based on split type
    if args.dataset == "refavs" and args.refavs_split == "test_n":
        logging.info({
            "S": eval_result['S']
        })
    else:
        logging.info({
            "miou": eval_result['miou'],
            "F_score": eval_result['F_score']
        })
    
if __name__ == "__main__":

    # Fix seed
    FixSeed = 42
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FixSeed)

    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="refavs", help="which subset of avsbench dataset")
    parser.add_argument("--output_mask_size", type=int, default=1024, help="Output mask size (None for original size)")
    parser.add_argument("--refavs_split", type=str, default="test_s")

    # Model parameters
    parser.add_argument("--save_path", type=str, default="./save_masks")
    # evaluation
    parser.add_argument("--save", action="store_true", help="Save the segmentation masks and ground truth")
    parser.add_argument("--weight_path", type=str, default="weights/model_best.pth", help="Path to the model weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model_name", type=str, default="primed")
    parser.add_argument("--one_second_avs_eval", action="store_true",
                        help="Evaluate every frame independently with its aligned 1-second audio token and fixed AVS text")
    
    # SAM2 model arguments
    parser.add_argument("--sam_scale", type=str, default="base", choices=["tiny", "small", "base", "large"], 
                        help="SAM2 model scale")
    parser.add_argument("--sam2_checkpoint", type=str, default="./pretrained/sam2_hiera_base_plus.pt")
    parser.add_argument("--semantic_flow_beta", type=float, default=1.0,
                        help="Cached-memory update strength in semantic flow")
    parser.add_argument("--prompt_injection_mode", type=str, default="injection",
                        choices=["injection", "prompt_encoder"],
                        help="How dense prompt features are injected")
    
    # Model components
    parser.add_argument("--num_compressed_tokens", type=int, default=4,
                        help="Number of compressed visual tokens shared across CBCF stages")

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    # AVS subsets freeze the audio cached-memory update; Ref-AVS keeps it enabled.
    args.disable_cached_memory = args.dataset in ("ms3", "s4")
    if args.disable_cached_memory:
        print("[AVS] disable_cached_memory=True: audio cached-memory update is frozen for AVS evaluation")
    main(args)
