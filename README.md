## An Efficient and Efficacious Network for Low-Light Image and Video Enhancement  



### Dataset

For preparing data and the pre-trained VGG19 model, [Li-Chongyi/UHDFour_code (github.com)](https://github.com/Li-Chongyi/UHDFour_code/tree/main?tab=readme-ov-file) repository provides detailed instructions.

### Environment

#### 1. Clone Repo

```
git clone <code_link>
cd EENet/
```

#### 2. Create Conda Environment and Install Dependencies

```
conda env create -f environment.yaml
conda activate EENet
```

### Quick Inference

Before performing the following steps, please download our pretrained model first.

The directory structure will be arranged as:

**Download Links:**链接：https://pan.baidu.com/s/1TJJdo9rLvpLIrpdr0H-jNw?pwd=ww12 提取码：ww12

```
ckpts
   |-LOLv1_checkpoint.pt  
   |-LOLv2_checkpoint.pt
```

Run the following command to process them:

```c++
CUDA_VISIBLE_DEVICES=X python src/test_PSNR.py --dataset-name our_test  
```

The enhanced images will be saved in the `results/` directory.

### Train

See `python3 src/train.py --h` for list of optional arguments, or `train.sh` for examples.

```
 CUDA_VISIBLE_DEVICES=X python src/train.py \
   --dataset-name LOLv1 \
   --train-dir ./data/LOL-v1/our485/\
   --valid-dir ./data/LOL-v1/eval15/ \
   --ckpt-save-path ./ckpts_training/ \
   --nb-epochs 1000 \
   --batch-size 6\
   --train-size 256 236 \
   --plot-stats \
   --cuda
```

### License

This codebase is released under the 

