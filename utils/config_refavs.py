from easydict import EasyDict as edict

"""
Ref-AVS dataset config
"""
cfg = edict()

###############################
# DATA
path_root = '/nvme0/home/public'
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = f"{path_root}/Ref-AVS/metadata.csv"
cfg.DATA.DIR_IMG = f"{path_root}/Ref-AVS/media"
cfg.DATA.DIR_MASK = f"{path_root}/Ref-AVS/gt_mask"
cfg.DATA.DIR_AUDIO_WAV = f"{path_root}/Ref-AVS/media"  # audio files are in media folder
cfg.DATA.NUM_FRAMES = 10  # number of frames per video
###############################
