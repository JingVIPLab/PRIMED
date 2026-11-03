import os
import pickle
import sys

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.resize_longest_side import ResizeLongestSide
from utils.config_m3 import cfg

sys.path.append("CLAP")

from msclap import CLAP


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_image_in_cv2_to_Tensor(path, mode='RGB', transform=None):
    image = cv2.imread(path)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class MS3Dataset_SAM(Dataset):
    """Dataset for multiple sound source segmentation"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(self, split, device=None):
        super(MS3Dataset_SAM, self).__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.split = split

        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform = ResizeLongestSide(1024)
        
        # Load CLAP model
        pretrained_path ="pretrained/CLAP_weights_2023.pth" 
        # Pass num_frames from config to CLAP
        self.clap_model = CLAP(
            model_fp=pretrained_path, 
            version='2023', 
            use_cuda=torch.device(self.device).type == "cuda",
            device=self.device,
            num_frames=cfg.DATA.NUM_FRAMES  # Read from config
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def _get_clap_embeddings(self, path, clap_embedding_path):
        # check if the embeddings are already saved
        clap_embedding_base_path = "/".join(clap_embedding_path.split("/")[:-1])
        if not os.path.exists(clap_embedding_base_path):
            os.makedirs(clap_embedding_base_path)
        
        if not os.path.exists(clap_embedding_path):
            with torch.no_grad():
                audio_embeddings = self.clap_model.get_audio_embeddings_single(path)
            audio_embeddings = audio_embeddings.detach().cpu()
            # save the embeddings
            with open(clap_embedding_path, 'wb') as fw:
                pickle.dump(audio_embeddings, fw)
        # load the embeddings
        else:
            with open(clap_embedding_path, 'rb') as fr:
                audio_embeddings = pickle.load(fr)
        return audio_embeddings

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        # video_name = df_one_video[0]
        video_name = df_one_video.iloc[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO_WAV, self.split, video_name + '.wav')
        
        clap_embedding_path = os.path.join(cfg.DATA.DIR_CLAP, self.split, video_name + '.pkl')
        sam_imgs, masks = [], []
        original_size_list = []
        
        # [p_A, p_V, p_AV] = [0.05, 0.05, 0.9]
        modality_prior_targets = torch.tensor([0.05, 0.05, 0.9], dtype=torch.float32)
        
        clap_embeddings = self._get_clap_embeddings(audio_wav_path, clap_embedding_path)
            
        for img_id in range(1, 6):
            # img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            img_path = os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id))

            # Load image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            original_size_list.append(image.shape[:2])

            # preprocess

            image = self.transform.apply_image(image)  # preprocess image for sam
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

            # append images
            sam_imgs.append(image)

        # load masks
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        
        sam_imgs_tensor = torch.stack(sam_imgs, dim=0)

        masks_tensor = torch.stack(masks, dim=0)
        
        return sam_imgs_tensor, clap_embeddings, "", masks_tensor, original_size_list, video_name, modality_prior_targets

    def __len__(self):
        return len(self.df_split)
