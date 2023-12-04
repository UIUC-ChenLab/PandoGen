# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash
#SBATCH --job-name="JOBNAME_gen_data"
#SBATCH --output="JOBNAME_gen_data.%j.%N.out"
#SBATCH --error="JOBNAME_gen_data.%j.%N.err"
#SBATCH --time=04:00:00
#SBATCH --export=ALL
<CPU_CONFIG>

<ENV_LOAD>

set -e -x -o pipefail

SCRIPTPATH=<SCRIPTPATH>
WORKDIR=<WORKDIR>
TSV=<TSV>
REF=<REF>
Timestamp=<Timestamp>
LAST_DATE=<LAST_DATE>
FIRST_DATE="<FIRST_DATE>"
first_date_opt=""
FASTA="<FASTA>"

[ $FASTA != '<FASTA>' ] && python $SCRIPTPATH/pandas_utils.py \
	--tsv $TSV \
	--fasta $FASTA \
	--protein "Spike" \
	--date_field "Submission date"


[ $FIRST_DATE != '<FIRST_DATE>' ] && first_date_opt="--first_date $FIRST_DATE"

python $SCRIPTPATH/random_split_fasta.py \
	--prefix $WORKDIR/decoder_data_unenumerated_${Timestamp} \
	--tsv $TSV \
	--last_date $LAST_DATE \
	--ref $REF \
	--n_train_per_bucket 9 \
	--n_val_per_bucket 1 \
	--n_test_per_bucket 0 \
	--protein Spike \
	--datefield "Submission date" $first_date_opt


cat $WORKDIR/decoder_data_unenumerated_${Timestamp}.train.mutations.lst $WORKDIR/decoder_data_unenumerated_${Timestamp}.val.mutations.lst > $WORKDIR/decoder_data_unenumerated_${Timestamp}.all.mutations.lst

ALL_SEQUENCES=$WORKDIR/decoder_data_unenumerated_${Timestamp}.all.mutations.lst

python $SCRIPTPATH/create_occurrence_buckets.py \
	--tsv $TSV \
	--availability_last_date $LAST_DATE \
	--datefield "Submission date" \
	--protein Spike \
	--period_length 7 \
	--output_prefix $WORKDIR/precomputed_competition_pairings_${LAST_DATE}_AsiaEurope_leading_weeks_${Timestamp} \
	--primary_locations "Asia,Europe" \
	--control_locations "~Asia,Europe" \
	--train_sequences $ALL_SEQUENCES \
	--val_sequences $WORKDIR/decoder_data_unenumerated_${Timestamp}.val.mutations.lst \
	--max_p_diff 0.25 \
	--combine_type union \
	--max_sequence_comparisons <N_COMP> \
	--max_to_verify 2000000 \
	--max_train 1000000 \
	--max_val 100000
