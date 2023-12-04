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

# Running PandoGen training

1. Downloadign a pretrained checkpoint

A pre-trained checkpoint on UniRef50 sequences is available at https://huggingface.co/oddjobs/pandogen-uda

PandoGen uses this checkpoint as a startpoint to perform finetuning.

2. On SLURM systems PandoGen supports push-button exeuction of the entire training pipeline. This can
be run as follows.

This command has been tested on both systems with V100 GPUs (16GB global memory) and A100 GPUs. Note that
some steps need 4 GPUs available at the same time.

```
python pandogen_train_top.py \
	--workdir <Working directory> \
	--tsv <variant_surveillance.tsv file from GISAID> \
	--last_date <Training cutoff date. Only data until this date is used> \
        --ref <A text file containing GISAID Spike protein reference> \
	--pretrained_ckpt <pandogen-uda model downloaded as above> \
        --name_prefix <Name prefixes for all SLURM jobs> \
        --first_date <If provided, only GISAID sequences reported after this date are used> \
        --finetune_batch_size <Batch size for SDA finetuning. Optional.> \
        --competition_batch_size <Batch size for reward model training. Optional.> \
        --quark_gen_batch_size <Batch size for data generation during the PandoGen finetuning step> \
        --quark_train_batch_size <Batch size for gradient descent> \
        --quark_gradient_checkpoint <Use gradient checkpointing> \
        --fasta <Spike fasta from GISAID for first time launch only> \
	--no_launch # Provide this option if SLURM commands only need to prepared, but not launched
```
Note that the parameters in `*.slurm` files in the `end_to_end_flow` directory will be automatically added
to the headers. Similarly the contents of the file `env_load.sh` will be inserted after the header to initialize
the environment. If these are different for your system, please keep the following files in your working directory with
the appropriate values: `gpu1_config.slurm`, `gpu4_config.slurm`, `cpu_config.slurm` and `env_load.sh`, and the script
will automatically pick those.

The training results will be in the directory `models` inside the working directory. The quark checkpoint will be
in `models/quark_JOBNAME_${Timestamp}` where JOBNAME is the option passed through `--name_prefix` and $Timestamp
is the time at which the job was launched.

3. Package PandoGen checkpoints

PandoGen checkpoints should be post-processed to be used in other machines or other locations. To do this, the following
script can be used:

```
python package_quark_model.py <PandoGen checkpoint> <Packaged Checkpoint>
```

# Running PandoGen sequence generation

```
 python predict_decoder.py \
                        --checkpoint <Final PandoGen model> \
                        --output_prefix <Prefix of output json file> \
                        --quark_model \
                        --gen_do_sample \
                        --gen_max_new_tokens 1398 \
                        --gen_num_return_sequences <Batch size> \
                        --gen_top_p <Top-p value> \
                        --num_batches <Number of batches to generate> \
                        --seed <Random seed>
```

If using the packaged checkpoint, please use the option  `--load_from_pretrained`.
