# TRAIN

## Dataset
Before preparing the dataset, you need to download our [HDRI lighting dataset](https://huggingface.co/datasets/yxuhan76/OpenDelight-Dataset/blob/main/HDRI.tar) to the `data` directory.

### Synthetic Dataset
We provide our rendered synthetic dataset [here](https://huggingface.co/datasets/yxuhan76/OpenDelight-Dataset/blob/main/ava256_aplit.tar).
You can directly download it to the `data` directory for use.

If you want to run the entire pipeline from the Light Stage scan dataset to image-albedo pairs by yourself, please refer to the following steps.

**Step 1:** download the Light Stage Scan dataset from [here](https://huggingface.co/datasets/yxuhan76/OpenDelight-Dataset/blob/main/ava256_4k.tar) and place it in the `data` directory. Note that following the same pipeline as described in our paper, we have processed the [Ava256 dataset](https://github.com/facebookresearch/ava-256) into the Light Stage scan format and made it open-source. We find that the Ava256 dataset has a larger scale and higher quality than NeRSemble, and models trained on it achieve better performance.

**Step 2:** Run the rendering script to render the Light Stage scan under random HDRI lighting and random viewpoints using Blender (our Light Stage scan dataset contains a total of 255 scans; each scan is rendered under 80 combinations of random lighting and viewpoints.):
```
cd synthetic
python batch_render.py \
    --data_root ../data/ava256_4k \
    --num_img 80 \
    --save_root ../workspace/ava256_raw
cd ..
```

**Step 3:** Align the rendered images according to the landmarks:
```
cd synthetic
python batch_align_render.py \
    --render_root ../workspace/ava256_raw \
    --scan_root ../data/ava256_4k
cd ..
```

**Step 4:** Split the dataset into training and validation sets.
```
cd synthetic
python split_train_val.py \
    --data_root ../workspace/ava256_raw \
    --save_root ../workspace/ava256_raw_split
cd ..
```

### FaceOLAT Dataset
Process the [FaceOLAT dataset](https://github.com/prraoo/FaceOLAT) into the format of `misc/faceolat_example`.
Note that when sampling random HDRI lighting to render OLAT images, you should also follow the **same sampling ratio** as in `synthetic/blender_render_scan.py`, i.e.:

```
HDRI_ROOT = "../data/HDRI"
HIGH_FREQ_PROB = 5
MED_FREQ_PROB = 3
LOW_FREQ_PROB = 1
```

We plan to contact the authors of FaceOLAT to release our fully processed complete FaceOLAT dataset.
Stay tuned.


## Train
### Base Delighting Network
Run the following command to start training.

```
CUDA_VISIBLE_DEVICE=0,1 python train.py \
    --config_path config/delight_base_network.yaml \
    --log_dir workspace/debug_ava256 \
    --batch_size 4 \
    --eval_freq 2000 \
    --ckpt_freq 50000 \
    --white_bg_color 0 \
    --aug_noise 0 \
    --syn_data_root PATH_TO_SYNTHETIC_DATASET \
    --real_data_root PATH_TO_FACEOLAT_DATASET \
    --syn_prob 0.5 \
    --ddp_port 12366 \
    --init_ckpt_path xxx \
    --grad_ckpt 1
```

We provide our own training logs and corresponding pre-trained weights [here](https://drive.google.com/file/d/15yR8ExmtvY3LdTTVbbUjQ3eg0l0949gZ/view?usp=sharing) for your reference, which were trained on 2 x NVIDIA RTX 3090 GPUs.

### Detail Enhancement Network
Run the following command to start training.

```
CUDA_VISIBLE_DEVICE=0,1 python ttrain_enhancer.py \
    --log_dir workspace/debug_ava256_enhancer \
    --batch_size 4 \
    --eval_freq 2000 \
    --ckpt_freq 50000 \
    --white_bg_color 0 \
    --mask_pho_loss 0 \
    --aug_noise 0 \
    --data_root PATH_TO_SYNTHETIC_DATASET \
    --use_normal 0 \
    --w_normal_loss 0.1 \
    --use_shadow_mask 0 \
    --w_shadow_mask 0.5 \
    --use_conf 0 \
    --conf_alpha 0.2 \
    --ddp_port 12355
```

We provide our own training logs and corresponding pre-trained weights [here](https://drive.google.com/file/d/1ct-DG5oaiwPiGD37CPZlp8o53BLYYPUH/view?usp=sharing) for your reference, which were trained on 2 x NVIDIA RTX 3090 GPUs.
