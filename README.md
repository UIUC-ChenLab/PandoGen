# PandoGen: Generating complete instances of future SARS-CoV-2 Spike protein using LLMs

PandoGen is a Protein Language Model aligned towards predicting
future viral sequences in the midst of a pandemic. The code in this repository can be used to
train models for the COVID-19 pandemic, forecasting Spike protein sequences.

Our pre-print is available at [biorxiv](https://www.biorxiv.org/content/10.1101/2023.05.10.540124v1)
Anand Ramachandran, Steven S. Lumetta, Deming Chen, "PandoGen: Generating complete instances of future SARS-CoV2 sequences using Deep Learning", bioRxiv 2023

# 1. System Requirements

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

# 2. Running PandoGen training

2.1. Downloading a pretrained checkpoint

A pre-trained checkpoint on UniRef50 sequences is available at [huggingface](https://huggingface.co/oddjobs/pandogen-uda)

PandoGen uses this checkpoint as a startpoint to perform finetuning.

2.2. On SLURM systems PandoGen supports push-button exeuction of the entire training pipeline. This can
be run as follows.

This command has been tested on both systems with V100 GPUs (16GB global memory) and A100 GPUs. Note that
some steps need 4 GPUs available at the same time.

```
python pandogen_train_top.py \
	--workdir <Working directory> \
	--tsv <variant_surveillance.tsv file from GISAID> \
	--last_date <Training cutoff date. Only data until this date is used to train the model> \
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
On non-SLURM system, the `--no_launch` option will prevent the launch, but produce command files that may be launched
sequentially by the user. These commands contain the input and output file names plumbed correctly from the script for
one training stage to the script for the next training stage. So the scripts only need to be launched using `bash <script.sh>` in
the correct sequence. The sequence for launching the scripts for manual launch is as follows:
1. `gen_data.sh`
2. `finetune.sh`
3. `train_competition.sh`
4. `quark_init.sh`
5. `validate_checkpoints.sh`
6. `quark_finetune.sh`

Note that the parameters in `*.slurm` files in the `end_to_end_flow` directory will be automatically added
to the headers. Similarly the contents of the file `env_load.sh` will be inserted after the header to initialize
the environment. If these are different for your system, please keep the following files in your working directory with
the appropriate values: `gpu1_config.slurm`, `gpu4_config.slurm`, `cpu_config.slurm` and `env_load.sh`. The `pandogen_train_top.py`
script will automatically pick the files in the working directory over the default files in the code repository in this case.

The training results will be in the directory `models` inside the working directory. The quark checkpoint will be
in `models/quark_JOBNAME_Timestamp` where JOBNAME is the option passed through `--name_prefix` and `Timestamp`
is the time at which the job was launched.

2.3. Package PandoGen checkpoints (Optional)

PandoGen checkpoints should be post-processed to be used in other machines or other locations. To do this, the following
script can be used:

```
python package_quark_model.py <PandoGen checkpoint> <Packaged Checkpoint>
```

# 3. Running PandoGen sequence generation

To sample sequences from the trained PandoGen model a single script `predict_decoder.py` can be used. The script simply
takes the trained checkpoint, and produces the output sequences. The various sampling parameters are passed to
the script as follows.

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
                        --seed <Random seed> \
			--load_from_pretrained  # This option is used if the PandoGen model is packaged as per step (2.3) above.
```

The output `JSON` file contains both generated sequences and their log-likelihoods, which may be used for ranking the sequences.
In case PandoGen is being trained on sequences known at some past point in the pandemic so as to validate it, the generated
output sequences may be compared with the sequences first reported after the training cutoff date in step 2.2. If PandoGen
is being trained using all GISAID sequences so as to use it for forecasting future sequences, simply known sequences may be
removed from the generated output file to get a list of new forecasts. Please follow our manuscript for details on how we
perfromed evaluations.
