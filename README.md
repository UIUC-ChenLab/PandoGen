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
