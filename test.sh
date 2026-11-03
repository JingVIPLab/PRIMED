python test.py \
  --dataset refavs \
  --refavs_split test_u \
  --weight_path ./weights/primed_refavs_best.pth \
  --sam_scale large \
  --sam2_checkpoint ./pretrained/sam2_hiera_large.pt
