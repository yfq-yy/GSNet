#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_model_size=1e6
max_layers=20
population_size=512
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for
resolution=32
epochs=1440

save_dir=../../save_dir/GhostShuffle_cifar_params2M_flops480M_maxlayer20_hs
mkdir -p ${save_dir}

horovodrun -np 4 -H localhost:4 python train_image_classification.py --dataset cifar10 --num_classes 10 \
  --dist_mode horovod --workers_per_gpu 6 --sync_bn \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 128 \
  --save_dir ${save_dir}/cifar10_1440epochs

# python train_image_classification.py --dataset cifar10 --num_classes 10 \
#   --dist_mode single --workers_per_gpu 6 \
#   --input_image_size 32 --epochs 1440 --warmup 5 \
#   --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#   --label_smoothing --random_erase --mixup --auto_augment \
#   --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
#   --arch Masternet.py:MasterNet \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt \
#   --batch_size_per_gpu 256 \
#   --save_dir ${save_dir}/cifar10_1440epochs
