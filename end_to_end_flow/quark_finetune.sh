# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash
#SBATCH --job-name="JOBNAME_quark_finetune"
#SBATCH --output="JOBNAME_quark_finetune.%j.%N.out"
#SBATCH --error="JOBNAME_quark_finetune.%j.%N.err"
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --dependency=afterok:<COMPETITION_VALIDATION>,afterok:<QUARK_INIT>
<GPU1_CONFIG>

<ENV_LOAD>

set -e -x -o pipefail

SCRIPTPATH=<SCRIPTPATH>
WORKDIR=<WORKDIR>
INIT_SEQUENCES=<INIT_SEQUENCES>
LOG_PREFIX=<LOG_PREFIX>
train_sequences=<TRAIN_SEQUENCES>
Timestamp=<Timestamp>
decoder_ckpt=<SDA>

logging_steps=4
warmup_steps=32
GEN_BATCH_SIZE=<QUARK_GEN_BATCH_SIZE>
GEN_BATCH_SIZE_INIT=32
N_EVAL_BATCHES=$((1024 / GEN_BATCH_SIZE))

UPDATE_SIZE=32
TRAIN_BATCH_SIZE=<QUARK_TRAIN_BATCH_SIZE>
n_acc=$((UPDATE_SIZE / TRAIN_BATCH_SIZE))

if [ $n_acc -eq 0 ]; then
        n_acc=1
fi

beta=0.1
prior_sequences=${train_sequences%.*}.prior_sequences.fa

if ! [[ -e $prior_sequences ]]; then
	grep -v "stop" $train_sequences > $prior_sequences
	python $WORKDIR/load_fa.py $prior_sequences
fi

potential_logs=$(ls ${LOG_PREFIX}*.err)
potential_model=$(python $WORKDIR/best_competition_ckpt.py $potential_logs)

gradient_checkpoint_args=<GRADIENT_CHECKPOINT_ARGS>

python $SCRIPTPATH/train_quark_finetune.py \
        --generative_model $decoder_ckpt \
        --potential_model $potential_model \
        --potential_pretrained_path $decoder_ckpt \
        --gen_num_return_sequences $GEN_BATCH_SIZE \
        --n_init_batches 0 \
	--init_sequences $INIT_SEQUENCES \
        --n_eval_batches $N_EVAL_BATCHES \
        --pool_size $((512 * 32)) \
        --prior_sequences $prior_sequences \
        --quantile_spec $WORKDIR/quantile_spec.json \
        --output_dir $WORKDIR/models/quark_JOBNAME_${Timestamp} \
        --gen_do_sample \
        --gen_max_new_tokens 1398 \
        --do_train \
        --evaluation_strategy epoch \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $n_acc \
        --learning_rate 5e-6 \
        --quark_beta $beta \
        --weight_decay 1e-2 \
        --num_train_epochs 24 \
        --lr_scheduler_type linear \
        --warmup_steps $warmup_steps \
        --logging_dir $WORKDIR/logs/quark_JOBNAME_${Timestamp}_logs \
        --logging_strategy steps \
        --logging_steps $logging_steps \
        --save_strategy epoch \
        --seed 13 \
        --fp16 \
        --dataloader_num_workers 4 \
	--no_dropout \
	--early_stopping 3 \
        --batched_membership_test \
        --membership_batch_size 16384 $gradient_checkpoint_args
