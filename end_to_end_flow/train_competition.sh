# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash
#SBATCH --job-name="JOBNAME_competition"
#SBATCH --output="JOBNAME_competition.%j.%N.out"
#SBATCH --error="JOBNAME_competition.%j.%N.err"
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --dependency=afterok:<FINETUNE>
<GPU4_CONFIG>

<ENV_LOAD>

set -e -x -o pipefail

SCRIPTPATH=<SCRIPTPATH>
WORKDIR=<WORKDIR>
train_pre=<TRAIN>
ref=<REF>
decoder_ckpt=<SDA>
Timestamp=<Timestamp>

train=${train_pre%.*}.no_stop.json
cat $train_pre | grep -v "stop" > $train
n_train=$(cat $train | wc -l)

UPDATE_SIZE=16
batch_size=<COMPETITION_BATCH_SIZE>
n_acc=$((UPDATE_SIZE / batch_size))
if [ $n_acc -eq 0 ]; then
        n_acc=1
fi

n_gpus=4
n_iters_per_epoch=$((n_train / batch_size / n_acc / n_gpus))
n_checkpoints_per_epoch=4
steps=$((n_iters_per_epoch / n_checkpoints_per_epoch))
logging_steps=4
warmup_steps=1024
master_port=$(shuf -i 20000-30000 -n 1)

torchrun --nproc_per_node $n_gpus --master_port $master_port \
$SCRIPTPATH/train_competition.py \
        --pretrained_path $decoder_ckpt \
        --ref $ref \
        --model_type Decoder \
        --precomputed_train_pairings $train \
        --precomputed_val_pairings $train \
        --attn_lr_deboost 1 \
        --referee_type binary \
        --prediction_type weight \
        --output_dir $WORKDIR/models/competition_JOBNAME_${Timestamp} \
        --do_train \
        --per_device_train_batch_size $batch_size \
        --gradient_accumulation_steps $n_acc \
        --learning_rate 1e-5 \
        --weight_decay 1e-2 \
        --num_train_epochs 1 \
        --lr_scheduler_type linear \
        --warmup_steps $warmup_steps \
        --logging_dir $WORKDIR/logs/compeition_JOBNAME_logs_${Timestamp} \
        --logging_strategy steps \
        --logging_steps $logging_steps \
        --save_strategy steps \
        --save_steps $steps \
        --seed 13 \
        --fp16 \
        --dataloader_num_workers 0
