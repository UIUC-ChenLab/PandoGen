# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash
#SBATCH --job-name="JOBNAME_competition_validation"
#SBATCH --output="JOBNAME_competition_validation.%j.%N.out"
#SBATCH --error="JOBNAME_competition_validation.%j.%N.err"
#SBATCH --time=4:00:00
#SBATCH --export=ALL
#SBATCH --dependency=afterok:<COMPETITION>
<GPU1_CONFIG>

<ENV_LOAD>

set -e -x

SCRIPTPATH=<SCRIPTPATH>
WORKDIR=<WORKDIR>
traindir=<TRAINDIR>
decoder_ckpt=<SDA>
ref=<REF>
TSV=<TSV>
Timestamp=<Timestamp>
LAST_DATE=<LAST_DATE>
# traindir=$(ls $WORKDIR/models/competition_lr_1e_5_2023* -d)
# decoder_ckpt=$(ls $WORKDIR/models/decoder_decoder_pretrain_2023* -d)
# ref=$HOME/COVID19/GISAID_COVID19_DATA/downloads__2021_06_30__2131/fasta/reference.txt

python $SCRIPTPATH/create_competition_validation_data.py \
        --tsv $TSV \
        --datefield "Submission date" \
        --protein "Spike" \
        --end_date $LAST_DATE \
        --ref $ref \
        --output_prefix $WORKDIR/data_eval_competition_${Timestamp}

for ckpt in $(ls $traindir/checkpoint-* -d); do
	output_prefix=$WORKDIR/$(basename $traindir)_$(basename $ckpt)

	python $SCRIPTPATH/run_validation_for_competition_models.py \
		--checkpoint_path $ckpt \
		--pretrained_path $decoder_ckpt \
		--target_sequences $WORKDIR/data_eval_competition_${Timestamp}.json \
		--output_prefix $output_prefix \
		--model_type Decoder \
		--predict_batch_size 7 \
		--embedding_batch_size 7
done
