#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_model_size=2e6
budget_flops=480e6
max_layers=20
population_size=512
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for


save_dir=../../save_dir/GhostShuffle_cifar_params2M_flops480M_shuffle_hs
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperGhostShuffleK3(8,16,1,8,1)SuperGhostShuffleK3(16,32,2,16,1)SuperGhostShuffleK3(32,64,2,32,1)SuperGhostShuffleK3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 0 \
  --zero_shot_score Zen \
  --fix_initialize \
  --origin \
  --search_space SearchSpace/search_space_ghostshuffle.py \
  --budget_model_size ${budget_model_size} \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size 32 \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 10 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 32 \
  --num_classes 10 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  > ${save_dir}/analyze_model.txt

# python train_image_classification.py --dataset cifar10 --num_classes 10 \
#   --dist_mode single --workers_per_gpu 6 \
#   --input_image_size 32 --epochs 1440 --warmup 5 \
#   --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#   --label_smoothing --random_erase --mixup --auto_augment \
#   --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
#   --arch Masternet.py:MasterNet \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt \
#   --batch_size_per_gpu 64 \
#   --save_dir ${save_dir}/cifar10_1440epochs


# python train_image_classification.py --dataset cifar100 --num_classes 100 \
#   --dist_mode single --workers_per_gpu 6 \
#   --input_image_size 32 --epochs 1440 --warmup 5 \
#   --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#   --label_smoothing --random_erase --mixup --auto_augment \
#   --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
#   --arch Masternet.py:MasterNet \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt \
#   --batch_size_per_gpu 64 \
#   --save_dir ${save_dir}/cifar100_1440epochs
