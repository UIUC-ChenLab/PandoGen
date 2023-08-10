# PandoGen: Generating complete instances of future SARS CoV2 Spike protein using LLMs

PandoGen is a Protein Language Model aligned towards predicting
future viral sequences in the midst of a pandemic. The code in this repository can be used to
train models for the COVID-19 pandemic, predicting Spike protein sequences.

Our pre-print is available at https://www.biorxiv.org/content/10.1101/2023.05.10.540124v1
Anand Ramachandran, Steven S. Lumetta, Deming Chen, "PandoGen: Generating complete instances of future SARS-CoV2 sequences using Deep Learning", bioRxiv 2023

# System Requirements

PandoGen has been tested on both PPC and Intel systems with V100 and A100 GPUs. PandoGen has the following dependencies

```
pysam
numpy
matplotlib
scipy
scikit-learn
pytorch (v1.10)
transformers (v4.16.2)
bioconda
biopython
```

# Running scripts

1. Generating fasta data from the variant\_surveillance.tsv file for finetuning SDA

```
OUTPUT_PREFIX=<prefix of output files>

python random_split_fasta.py \
        --prefix $OUTPUT_PREFIX \
        --tsv /path/to/variant_surveillance.tsv \
        --last_date $LAST_DATE \
        --ref <SARS-CoV2 reference sequence> \
        --n_train_per_bucket <Number of training buckets> \
        --n_val_per_bucket <Number of validation buckets> \
        --n_test_per_bucket <Number of test buckets> \
        --protein Spike \
        --datefield "Submission date"
```

Here,`--last\_date` represents the last date of the training period.

2. Generating reward modeling data

```
# Note the previous train/val spiit is not used in the following.
# We still pass a dummy value for -val\_sequences
cat ${OUTPUT\_PREFIX}.train.mutations.lst ${OUTPUT\_PREFIX}.val.mutations.lst > $ALL\_MUTATIONS

python create_occurrence_buckets.py \
        --tsv $TSV \
        --availability_last_date $LAST_DATE \
        --datefield "Submission date" \
        --protein Spike \
        --period_length 7 \
        --output_prefix $REWARD_MODEL_DATA_PREFIX \
        --primary_locations "Asia,Europe" \
        --control_locations "~Asia,Europe" \
        --train_sequences $ALL_SEQUENCES \
        --val_sequences ${OUTPUT\_PREFIX}.val.mutations.lst \
        --max_p_diff <set a max diff value> \
        --combine_type union \
        --max_sequence_comparisons <Set a max limit> \
        --max_to_verify <set a mx limit value> \
        --max_train <Set a max limit value> \
        --max_val <Set a max limit value>
```

3. Create SDA model by finetuning on SARS-CoV2 data

```
torchrun --nproc\_per\_node $N\_GPUS train\_decoder.py \
                --train $train \
                --val $val \
                --output_dir $SDA_PATH \
                --do_train \
                --per_device_train_batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $N_ACC \
                --learning_rate $LR \
                --num_train_epochs $N_EPOCHS \
                --lr_scheduler_type linear \
                --warmup_steps $WARMUP_STEPS \
                --weight_decay $WEIGHT_DECAY \
                --log_level info \
                --logging_strategy steps \
                --logging_first_step  \
                --logging_steps $LOGGING_STEPS \
                --save_strategy steps \
                --save_steps $SAVE_STEPS \
                --evaluation_strategy steps \
                --eval_steps $SAVE_STEPS \
                --fp16 \
                --dataloader_num_workers 0 \
                --intermediate_size $FC_SIZE \
                --hidden_size $ATTN_SIZE \
                --num_hidden_layers 8 \
                --num_attention_heads $N_HEADS \
                --max_position_embeddings $N_MAX_LENGTH \
                --checkpoint_params $ckpt \
                --checkpoint_model_type "Decoder" \
                --num_quark_quantiles 11 \
                --load_best_model_at_end
```

4. Train reward model

```
torchrun --nproc_per_node $N_GPUS train_competition.py \
        --pretrained_path $SDA_PATH \
        --ref <Reference sequence> \
        --model_type Decoder \
        --precomputed_train_pairings <Data from (1)> \
        --precomputed_val_pairings <Data from (1)> \
        --attn_lr_deboost 1 \
        --referee_type binary \
        --prediction_type weight \
        --output_dir $REWARD_OUTPUT_PATH \
        --do_train \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $N_ACC \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --num_train_epochs 1 \
        --lr_scheduler_type linear \
        --warmup_steps $warmup_steps \
        --logging_strategy steps \
        --logging_steps $LOGGING_STEPS \
        --save_strategy steps \
        --save_steps $SAVE_STEPS \
        --fp16 \
        --dataloader_num_workers 0
```

5. Validate and select best reward model


```
# Create validation data
python create_competition_validation_data.py \
        --tsv $TSV \
        --datefield "Submission date" \
        --protein "Spike" \
        --end_date $LAST_DATE \
        --ref $ref \
        --output_prefix $REWARD_VALIDATION_PREFIX

for ckpt in $(ls $REWARD_OUTPUT_PATH/checkpoint-* -d); do
        output_prefix=<Output of evaluation results>

        python /home/aramach4/COVID19/covid19/run_validation_for_competition_models.py \
                --checkpoint_path $ckpt \
                --pretrained_path $SDA_PATH \
                --target_sequences ${REWARD_VALIDATION_PREFIX}.json \
                --output_prefix $output_prefix \
                --model_type Decoder \
                --predict_batch_size 7 \
                --embedding_batch_size 7
done
```

6. Initialize PandoGen finetuning by generating an initial sequence set

```
python predict_decoder.py \
        --checkpoint $decoder_ckpt \
        --output_prefix ${INIT_SEQUENCES%.*} \
        --gen_do_sample \
        --gen_max_new_tokens 1398 \
        --gen_num_return_sequences <Batch size> \
        --num_batches <Number of batches>
```

7. Perform PandoGen finetuning

```
python train_quark_finetune.py \
        --generative_model $SDA_PATH \
        --potential_model <Best checkpoint from step 5> \
        --potential_pretrained_path $SDA_PATH \
        --gen_num_return_sequences $GEN_BATCH_SIZE \
        --n_init_batches 0 \
        --init_sequences $INIT_SEQUENCES \
        --n_eval_batches $N_EVAL_BATCHES \
        --pool_size <Size of quark data pool> \
        --prior_sequences <Training sequences> \
        --quantile_spec <Quantile specification> \
        --output_dir <Results directory> \
        --gen_do_sample \
        --gen_max_new_tokens 1398 \
        --do_train \
        --evaluation_strategy epoch \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $n_acc \
        --learning_rate <Learning rate> \
        --quark_beta <Quark beta parameter> \
        --weight_decay <Weight decay> \
        --num_train_epochs <Number of epochs> \
        --lr_scheduler_type linear \
        --warmup_steps $WARMUP_STEPS \
        --logging_strategy steps \
        --logging_steps $LOGGING_STEPS \
        --save_strategy epoch \
        --fp16 \
        --dataloader_num_workers 4 \
        --no_dropout \
        --early_stopping <Number of early stopping steps>
```
