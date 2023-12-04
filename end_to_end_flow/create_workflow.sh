# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#!/bin/bash

module load opence/1.6.1

if [ $# -eq 0 ]; then
	echo "Usage: $0 <LAST_DATE> <WORKDIR> [TSV (optional)] [#competitions per sequence (optional)]"
	exit
fi

set -e -x -o pipefail

LAST_DATE=$1
WORKDIR=$2
TSV=""
N_COMP=750

if [ $# -gt 2 ]; then
	TSV=$3
fi

if [ $# -gt 3 ]; then
	N_COMP=$4
fi

SCRIPTPATH=/home/aramach4/COVID19/UniProt_2022_Oct_13_04_00_17_CDT/UniRef_download_2022_Oct_13_04_00_17_CDT/uniref50/end_to_end_flow

mkdir -p $WORKDIR

cp $SCRIPTPATH/* $WORKDIR/

for i in `ls $WORKDIR/*.sh`; do
	sed "s?<WORKDIR>?$WORKDIR?g" -i $i
	sed "s?<LAST_DATE>?$LAST_DATE?g" -i $i
	if [ $(echo $TSV | wc | awk '{print $3 - 1}') -gt 0 ]; then
		sed "s?TSV=.*?TSV=$TSV?g" -i $i
	fi
	sed "s?<N_COMP>?$N_COMP?g" -i $i
done

order=(
finetune.sh
train_competition_lr1e_5.sh
validate_checkpoints_1e_5.sh
quark_init.sh
quark_finetune_beta_0.1_no_dropout_no_stop_24_epochs.sh
)

cd $WORKDIR
sub=$(swbatch gen_data.sh | grep "Submitted batch job")
job_id=$(echo $sub | sed 's/Submitted batch job //g')

for o in ${order[@]}; do
	sed "s/<JOB_ID>/$job_id/g" -i $o
	sub=$(swbatch $o | grep "Submitted batch job")
	job_id=$(echo $sub | sed 's/Submitted batch job //g')
done
