<h1 align="center">
 [ICCV 2025 Highlight] <i>VRM: Knowledge Distillation via Virtual Relation Matching</i> </h1>
<p align="center">

This repository contains a PyTorch implementation of **Virtual Relation Matching (VRM)** introduced in the paper 
<i>VRM: Knowledge Distillation via Virtual Relation Matching (ICCV 2025)</i>.


## Preparation

1. Clone the repository to your local workspace:

    ```
    git clone https://github.com/VISION-SJTU/VRM.git
    ```

2. Configure the environment:
    ```
    conda create --name vrm python=3.8
    conda activate vrm
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    python setup.py develop
    ```
    Note that other torch versions may also work.


3. Prepare the data
   
    The CIFAR-100 dataset will be automatically downloaded to `./data/cifar100/`. The ImageNet dataset shall be manually downloaded to `./data/imagenet/`. Pretrained teachers on CIFAR-100 can be downloaded from <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and should be unzipped to `./download_ckpts/cifar_teachers`.


## Training

To train on CIFAR-100, you may use:

```bash
  # resnet32x4-to-resnet8x4 distillation as an example
  python tools/train_vrm.py --cfg configs/cifar100/vrm/res32x4_res8x4.yaml
 ```

To train on ImageNet, you may use:
```bash
  # resnet34-to-resnet18 distillation as an example
  python tools/train_vrm.py --cfg configs/imagenet/r34_r18/vrm.yaml
 ```

To train on ImageNet using multiple GPUs, you may use:

```bash
  # resnet34-to-resnet18 distillation trained on 4 GPUs as an example
  python -m torch.distributed.launch --nproc_per_node=4 tools/train_vrm_dist.py --cfg configs/imagenet/r34_r18/vrm.yaml
 ```

## Evaluation

The distilled student model will be automatically evaluated on the validation set during training. Manual evaluation is also supported by using `eval.py`.


## Acknowledgement

This project is developed using the [timm](https://github.com/rwightman/pytorch-image-models) and the [mdistiller](https://github.com/megvii-research/mdistiller) library.

## Reference

If you find this project useful, please consider citing it:

```
@inproceedings{zhang2025rsd,
  author    = {Weijia Zhang and Fei Xie and Weidong Cai and Chao Ma},
  title     = {VRM: Knowledge Distillation via Virtual Relation Matching},
  booktitle = {ICCV},
  year      = {2025}
}
```
# test
