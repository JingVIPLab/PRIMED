import os
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.config_refavs import cfg

sys.path.append("CLAP")

from msclap import CLAP


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    """Load image and convert to tensor.
    Args:
        path: Image file path
        mode: PIL image mode (default: 'RGB')
        transform: Optional torchvision transform 
    Returns:
        img_tensor: Image as tensor
        original_size: [height, width] of original image
    """
    img_PIL = Image.open(path).convert(mode)
    w, h = img_PIL.size
    
    if transform:
        img_tensor = transform(img_PIL)
    else:
        # If no transform provided, convert to tensor manually
        img_tensor = transforms.ToTensor()(img_PIL)
    
    return img_tensor, [h, w]


class RefAVSDataset_SAM(Dataset):
    """Dataset for Referring Audio-Visual Segmentation"""

    def __init__(self, split, device=None):
        super(RefAVSDataset_SAM, self).__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.max_size = 1024
        self.split = split
        self.num_frames = cfg.DATA.NUM_FRAMES

        # Load metadata
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))

        self.img_transform = transforms.Compose([
            transforms.Resize([self.max_size, self.max_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([self.max_size, self.max_size]),
            transforms.ToTensor()
        ])

        # Load CLAP model
        pretrained_path = "pretrained/CLAP_weights_2023.pth"
        # Pass num_frames from config to CLAP
        self.clap_model = CLAP(
            model_fp=pretrained_path, 
            version='2023', 
            use_cuda=torch.device(self.device).type == "cuda",
            device=self.device,
            num_frames=cfg.DATA.NUM_FRAMES  # Read from config
        )
        
        # Note: Text encoder moved to model (following SAMWISE approach)
        # Text tokenization and encoding with automatic padding is handled in the model's forward pass
        # This avoids variable-length sequence issues in DataLoader collate_fn
        
    def _get_clap_embeddings(self, path):
        """Get CLAP embeddings without caching - generate on-the-fly."""
        with torch.no_grad():
            audio_embeddings = self.clap_model.get_audio_embeddings_single(path)
        audio_embeddings = audio_embeddings.detach().cpu()
        return audio_embeddings

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        uid = df_one_video['uid']
        fid = df_one_video['fid']
        exp = df_one_video['exp']
        
        api_result = df_one_video.get('api_result', None)
        # Shape: [p_A, p_V, p_AV]
        if api_result is not None:
            api_result = str(api_result).strip().lower()
            if 'visual' in api_result and 'audio' not in api_result:
                # visual only
                modality_prior_targets = torch.tensor([0.05, 0.9, 0.05], dtype=torch.float32)
            elif 'audio' in api_result and 'visual' not in api_result:
                # audio only
                modality_prior_targets = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float32)
            else:
                # audio-visual (default)
                modality_prior_targets = torch.tensor([0.05, 0.05, 0.9], dtype=torch.float32)
        else:

            modality_prior_targets = torch.tensor([0.05, 0.05, 0.9], dtype=torch.float32)
        
        # Extract video id from uid
        vid = uid.rsplit('_', 2)[0]

        # Paths
        img_base_path = os.path.join(cfg.DATA.DIR_IMG, vid, 'frames')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, vid, f'fid_{fid}')
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO_WAV, vid, 'audio.wav')

        sam_imgs, masks = [], []
        original_size_list = []

        # Get CLAP embeddings (on-the-fly, no caching)
        clap_embeddings = self._get_clap_embeddings(audio_wav_path)
        
        # Text will be processed in model (using exp directly)
        # Following SAMWISE: return raw text string, model handles tokenization + encoding + padding

        # Load frames and masks
        for frame_idx in range(self.num_frames):
            # Load image
            img_path = os.path.join(img_base_path, f'{frame_idx}.jpg')
            image, original_size = load_image_in_PIL_to_Tensor(img_path, transform=self.img_transform)
            original_size_list.append(original_size)
            sam_imgs.append(image)

            # Load mask
            mask_filename = f'{frame_idx:05d}.png'
            mask_path = os.path.join(mask_base_path, mask_filename)
            mask, _ = load_image_in_PIL_to_Tensor(mask_path, transform=self.mask_transform, mode='1')
            masks.append(mask)

        sam_imgs_tensor = torch.stack(sam_imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        # Return exp (raw text) instead of text_embeddings + modality_prior_targets
        # Model will handle tokenization and encoding with automatic padding
        # Only return modality_prior_targets for train/val splits, not for test
        if self.split.startswith('test'):
            return sam_imgs_tensor, clap_embeddings, exp, masks_tensor, original_size_list, vid, fid
        else:
            return sam_imgs_tensor, clap_embeddings, exp, masks_tensor, original_size_list, vid, fid, modality_prior_targets

    def __len__(self):
        return len(self.df_split)
