from easydict import EasyDict as edict

"""
default config
"""
cfg = edict()
###############################
# DATA
path_root = '/nvme0/home/public'
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = f"{path_root}/avsbench_data/Multi-sources/ms3_meta_data.csv"
cfg.DATA.DIR_IMG = f"{path_root}/avsbench_data/Multi-sources/ms3_data/visual_frames"
cfg.DATA.DIR_AUDIO_WAV = f"{path_root}/avsbench_data/Multi-sources/ms3_data/audio_wav"
cfg.DATA.DIR_MASK = f"{path_root}/avsbench_data/Multi-sources/ms3_data/gt_masks"
cfg.DATA.DIR_CLAP = f"{path_root}/avsbench_data/Multi-sources/ms3_data/clap_embeddings"
cfg.DATA.NUM_FRAMES = 5  # number of frames per video
###############################
