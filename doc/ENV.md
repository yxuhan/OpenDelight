# Environment

Download the [pretrained MAE weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) into `pretrained/mae/mae_visualize_vit_base.pth`.

First, install the basic environment as follows: 

```
conda create -n od python=3.10
conda activate od

# install pytorch
pip install https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu121/torchvision-0.18.1%2Bcu121-cp310-cp310-linux_x86_64.whl

# install pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2

# install blenderproc
pip install blenderproc==2.8.0
then, run 'blenderproc quickstart' in the terminal for testing environment
(see https://github.com/DLR-RM/BlenderProc)

# install other libs
pip install tqdm \
    trimesh \
    lpips \
    pathspec \
    tensorboard \
    timm \
    setuptools==44.0.0 \
    pyfacer \
    onnx==1.18.0 \
    onnxruntime-gpu==1.22.0
```

Then, install ibug face-raleted libs:

```
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

Lastly, download the [weights](https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitl16_384.onnx) of the Soft Foreground Segmentation model from [DAViD](https://github.com/microsoft/DAViD), then put it to `matting/model`.
