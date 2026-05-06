#!/bin/bash

if [ -z "$1" ]
then
	blr=2e-4
else
	blr=$1
fi

if [ -z "$2" ]
then
	mask_t_prob=0.2
else
	mask_t_prob=$2
fi

if [ -z "$3" ]
then
	mask_f_prob=0.2
else
	mask_f_prob=$3
fi

if [ -z "$4" ]
then
	ckpt= /path/to/checkpoint
else
	ckpt=$4
fi

source /path/to/env



audioset_train_all_json=/path/to/data_train
audioset_eval_json=/path/to/data_eval


dataset=iemocap

# export CUDA_VISIBLE_DEVICES=3

python -m torch.distributed.launch --nproc_per_node=4 --use_env main_finetune_as.py \
    --model vit_base_patch16 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --data_eval $audioset_eval_json \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 4 \
    --roll_mag_aug True \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 \
    --epochs 200 \
    --warmup_epochs 10 \
    --replacement False \
    --distributed_wrapper False \
    --mask_2d True \
    --output_dir "output_ft_dir" \
    --log_dir "output_ft_dir/logs" \
    --nb_classes 4 \
    --mixup 0 \

