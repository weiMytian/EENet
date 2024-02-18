 CUDA_VISIBLE_DEVICES=0 python src/train.py \
   --dataset-name LOLv1 \
   --train-dir ./data/LOL-v1/our485/\
   --valid-dir ./data/LOL-v1/eval15/ \
   --ckpt-save-path ./ckpts_training/ \
   --nb-epochs 1000 \
   --batch-size 6\
   --train-size 256 236 \
   --plot-stats \
   --cuda

# CUDA_VISIBLE_DEVICES=0 python src/train.py \
#   --dataset-name LOLv2 \
#   --train-dir ./data/LOL-v2/train/\
#   --valid-dir ./data/LOL-v2/test/ \
#   --ckpt-save-path ./ckpts_training/ \
#   --nb-epochs 1000 \
#   --batch-size 6\
#   --train-size 256 256 \
#   --plot-stats \
#   --cuda    
