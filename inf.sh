#!/bin/bash
#SBATCH --job-name=aud-ft
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=480GB
#SBATCH --signal=USR1@120
#SBATCH --output=/checkpoint/%u/jobs/%A.out
#SBATCH --error=/checkpoint/%u/jobs/%A.err

source /train20/sppro/permanent/cqchen5/AudioMAE/bin/activate

audioset_train_json=/train20/sppro/permanent/cqchen5/datasets/IEMOCAP/origin_data/fold_1/4class/tsv_files/train.tsv

audioset_eval_json=/train20/sppro/permanent/cqchen5/datasets/IEMOCAP/origin_data/fold_1/4class/tsv_files/valid.tsv

# audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv

dataset=iemocap

if [ -z "$1" ]
then
    ckpt=''
else
    ckpt=$1
fi


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_as.py \
--log_dir ./output_infer_dir/log_dir \
--output_dir ./output_infer_dir \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--finetune $ckpt \
--batch_size 8 \
--eval \



