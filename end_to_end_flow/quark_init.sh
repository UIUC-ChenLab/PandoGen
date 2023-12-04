# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash
#SBATCH --job-name="JOBNAME_quark_init"
#SBATCH --output="JOBNAME_quark_init.%j.%N.out"
#SBATCH --error="JOBNAME_quark_init.%j.%N.err"
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --dependency=afterok:<FINETUNE>
<GPU1_CONFIG>

<ENV_LOAD>

set -e -x -o pipefail

SCRIPTPATH=<SCRIPTPATH>
WORKDIR=<WORKDIR>
Timestamp=<Timestamp>
decoder_ckpt=<SDA>

GEN_BATCH_SIZE_INIT=32
INIT_SEQUENCES=$WORKDIR/quark_init_${Timestamp}.json

python $SCRIPTPATH/predict_decoder.py \
        --checkpoint $decoder_ckpt \
        --output_prefix ${INIT_SEQUENCES%.*} \
        --gen_do_sample \
        --gen_max_new_tokens 1398 \
        --gen_num_return_sequences $GEN_BATCH_SIZE_INIT \
        --num_batches 512
