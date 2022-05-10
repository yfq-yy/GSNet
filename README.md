# GSNet

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.zen_nas?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/latest?definitionId=5&branchName=main)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/38905)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

Our based code is forked from [ZenNAS](https://github.com/idstcv/ZenNAS).
 We modify the code to make it suitable for UAV embedding embedded platforms.
 
We mainly made the following changes:

- redesign a new lightweight search space, GhostShuffle unit(GS unit), which further reduce the params and FLOPs of network.
- add constraints on params, FLOPs ,layers and MAC, and searches for the optimal network GSmodel-L based on the
ZenScore scores.

## Experimental results

We tested the modified code and verified its correctness. The results are as follows:

We used apex with mixed precision to complete the training within 5 days on 4 2080Ti GPUs,
and the results are consistent with the paper.

|    dataset   | paper model params   |   paper model FLOPs   |     mAP        |
| :----------: | :------------------: | :-------------------: |:---------------|
|  VisDrone    |        6.61M         |       11.20M          |    14.92%      |
| UAV-OUC-DET  |        6.61M         |       11.20M          |    8.38%       |




## Reproduce Paper Experiments

### System Requirements

- PyTorch = 1.8.0, Python = 3.7.9, CUDA=10.2
- By default, ImageNet dataset is stored under \~/data/imagenet;
CIFAR-10/CIFAR-100 is stored under \~/data/pytorch\_cifar10 or \~/data/pytorch\_cifar100
- Pre-trained parameters are cached under \~/.cache/pytorch/checkpoints/zennet\_pretrained

### Package Requirements

- ptflops
- opencv = 4.50 
- torchvision = 0.9.0
- tensorboard >= 1.15 (optional)
- apex

### Evaluate pre-trained models on ImageNet and CIFAR-10/100

To evaluate the pre-trained model on ImageNet using GPU 0:

``` bash
python val.py --fp16 --gpu 0 --arch ${zennet_model_name}
```

where ${zennet\_model\_name} should be replaced by a valid ZenNet model name.
The complete list of model names can be found in the 'Pre-trained Models' section.

To evaluate the pre-trained model on CIFAR-10 or CIFAR-100 using GPU 0:

``` bash
python val_cifar.py --dataset cifar10 --gpu 0 --arch ${zennet_model_name}
```

To create a ZenNet in your python code:

``` python
gpu=0
model = ZenNet.get_ZenNet(opt.arch, pretrained=True)
torch.cuda.set_device(gpu)
torch.backends.cudnn.benchmark = True
model = model.cuda(gpu)
model = model.half()
model.eval()
```

### usage

We supply apex and Horovod distributed training scripts, you can modify other original scripts based on these scripts.
apex script:

```bash
scripts/Zen_NAS_ImageNet_latency0.1ms_train_apex.sh
```

Horovod script:

```bash
scripts/Zen_NAS_ImageNet_latency0.1ms_train.sh
```

If you want to search model, please notice the choices "--fix_initialize" and "--origin".
"--fix_initialize" decides how to initialize population, the algorithm default choice is random initialization.
"--origin" determines how the mutation model is generated.  
When specified "--origin", the mutated model will be produced using the original method.

### Searching on CIFAR-10/100

Searching for CIFAR-10/100 models with budget params < 1M, using different zero-shot proxies:

```bash
scripts/Flops_NAS_cifar_params1M.sh
scripts/GradNorm_NAS_cifar_params1M.sh
scripts/NASWOT_NAS_cifar_params1M.sh
scripts/Params_NAS_cifar_params1M.sh
scripts/Random_NAS_cifar_params1M.sh
scripts/Syncflow_NAS_cifar_params1M.sh
scripts/TE_NAS_cifar_params1M.sh
scripts/Zen_NAS_cifar_params1M.sh


### Copyright

Copyright 2021 ZTE corporation. All Rights Reserved.


