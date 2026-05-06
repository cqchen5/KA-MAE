#!/bin/bash
if [ -z "$1" ]
then
    blr=2e-4
else
    blr=$1
fi

source /path/to/env


audioset_train_all_json=/path/to/data_train


dataset=iflytek


OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
--batch_size 20 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.9 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 1 \
--blr $blr --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--roll_mag_aug True \
--decoder_mode 1 \
--output_dir "output_dir" \
--log_dir "output_dir/logs" \