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



For proving the effectiveness of the algorithm, we experimented with several different model searches
and get the following result.
We use a single Tesla V100 GPU to evolve the population 50000 times.
|    method     |    model    | search time(hours) | model score |
| :-----------: | :---------: | :----------------: | :---------: |
|    ZenNAS     | latency01ms |      98.4274       |   126.038   |
|        \       | latency05ms |      22.0189       |   243.101   |
|        \       | latency08ms |      28.5952       |   304.323   |
|        \       | latency12ms |      44.6237       |   375.027   |
| modify-ZenNAS | latency01ms |       64.988       |   134.896   |
|        \       | latency05ms |      20.9895       |   245.712   |
|        \       | latency08ms |      25.0358       |   310.629   |
|        \       | latency12ms |       43.239       |   386.669   |

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
```


## Customize Your Own Search Space and Zero-Shot Proxy

The masternet definition is stored in "Masternet.py".
The masternet takes in a structure string and parses it into a PyTorch nn.Module object.
The structure string defines the layer structure which is implemented in "PlainNet/*.py" files.
For example, in "PlainNet/SuperResK1KXK1.py",
we defined SuperResK1K3K1 block, which consists of multiple layers of ResNet blocks.
To define your block, e.g. ABC_Block, first, implement "PlainNet/ABC_Block.py".
Then in "PlainNet/\_\_init\_\_.py",  after the last line, append the following lines to register the new block definition:

```python
from PlainNet import ABC_Block
_all_netblocks_dict_ = ABC_Block.register_netblocks_dict(_all_netblocks_dict_)
```

After the above registration call, the PlainNet module can parse your customized block from the structure string.

The search space definitions are stored in SearchSpace/*.py. The important function is

```python
gen_search_space(block_list, block_id)
```

block_list is a list of super-blocks parsed by the masternet.
block_id is the index of the block in block_list which will be replaced later by a mutated block
This function must return a list of mutated blocks.

### Direct specify search space

"PlainNet/AABC_Block.py" has defined the candidate blocks,
you can directly specify candidate blocks in the search spaces by passing parameters "--search_space_list".
So you have two methods to specify search spaces.
Taking ResNet-like search space as an example, you can use "--search_space SearchSpace/search_space_XXBL.py" or
"--search_space_list PlainNet/SuperResK1KXK1.py PlainNet/SuperResKXKX.py" to specify search space. Both of them are equivalent.

In scripts, when you choose to use the first method to specify search space,
**you should also add other two parameters "--fix_initialize" and"--origin"**,
so the algorithm will initialize with a fixed model.

The zero-shot proxies are implemented in "ZeroShotProxy/*.py". The evolutionary algorithm is implemented in "evolution_search.py".
"analyze_model.py" prints the FLOPs and model size of the given network.
"benchmark_network_latency.py" measures the network inference latency.
"train_image_classification.py" implements SGD gradient training
and "ts_train_image_classification.py" implements teacher-student distillation.


## Copyright

Copyright 2021 ZTE corporation. All Rights Reserved.
