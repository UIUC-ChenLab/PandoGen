# PandoGen: Generating complete instances of SARS CoV2 Spike protein using LLMs

PandoGen does reward-based finetuning of a Protein Language Model to align it towards predicting
protein sequences in the midst of a pandemic. The application is to SARS-CoV2 Spike protein sequences.

In the midst of a pandemic, there are two requirements from a sequence generator
* Generate *novel* sequences as opposed to sequences from the training set
* Generate *salient* sequences, which cause multiple infections rather than those with trivial case counts

PandoGen achieves both of these goals through a novel reward model without requiring expensive annotations
or laboratory characterizations.

Our pre-print is available at https://www.biorxiv.org/content/10.1101/2023.05.10.540124v1
Anand Ramachandran, Steven S. Lumetta, Deming Chen, "PandoGen: Generating complete instances of future SARS-CoV2 sequences using Deep Learning", bioRxiv 2023
