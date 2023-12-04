# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
"""
Top-level script to train PandoGen. The method uses SLURM commands.
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import re
from collections import namedtuple
import shutil
import subprocess
import datetime
from transformers import HfArgumentParser

_THISDIR = os.path.split(os.path.abspath(__file__))[0]


GenDataResults = namedtuple(
    "GenDataResults",
    [
        "script",
        "jobid",
        "generator_train_fa",
        "generator_val_fa",
        "competition_train_json",
    ]
)

FinetuneResults = namedtuple(
    "FinetuneResults",
    [
        "script",
        "jobid",
        "finetune_model_path",
    ]
)

CompetitionRawResults = namedtuple(
    "CompetitionRawResults",
    [
        "script",
        "jobid",
        "checkpoint_storage",
    ]
)

ValidationResults = namedtuple(
    "ValidationResults",
    [
        "script",
        "jobid",
        "validation_log_prefix",
    ]
)

QuarkInitResults = namedtuple(
    "QuarkInitResults",
    [
        "script",
        "jobid",
        "init_file",
    ]
)

QuarkFinetuneResults = namedtuple(
    "QuarkFinetune",
    [
        "script",
        "jobid",
        "model_path",
    ]
)


def full_path(f: str) -> str:
    fname = os.path.join(_THISDIR, "end_to_end_flow", f)
    if os.path.exists(fname):
        return fname

    raise ValueError(f"Unknwon file {f} or {fname}")


@dataclass
class Arguments:
    workdir: str = field(
        metadata={"help": "Working directory path"}
    )

    tsv: str = field(
        metadata={"help": "variants tsv file from GISAID"}
    )

    last_date: str = field(
        metadata={"help": "Last date (inclusive) of data to use for training"}
    )

    ref: str = field(
        metadata={"help": "Reference sequence file"}
    )

    pretrained_ckpt: str = field(
        metadata={"help": "Pretrained checkpoint path"}
    )

    max_sequence_comparisons: int = field(
        default=750,
        metadata={"help": "Maximum number of comparisons per sequence for game-play"}
    )

    name_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Prefix for a job name (for SLURM)"}
    )

    first_date: Optional[str] = field(
        default=None,
        metadata={"help": "First date (inclusive) of data after which to use sequences for training"}
    )

    finetune_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for finetuning"}
    )

    competition_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for competition training"}
    )

    quark_gen_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for quark generation step"}
    )

    quark_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for quark gradient descent"}
    )

    quark_gradient_checkpoint: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing for quark"}
    )

    fasta: str = field(
        default=None,
        metadata={"help": "If provided, do a preprocessing run for TSV"}
    )

    no_launch: bool = field(
        default=False,
        metadata={"help": "Do not launch jobs, simply create commands (for debug)"}
    )

    def __post_init__(self):
        self.timestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S") + "_UTC"
        self.launch = not self.no_launch
        self.header_fixer = HeaderReplace(self.workdir)


def replace_helper(src: str, replace_dict: dict) -> str:
    res = str(src)

    for key, value in replace_dict.items():
        res = res.replace(key, str(value))

    return res    


def launch_job(job_script: str, launcher: str = "sbatch"):
    """
    Launch a job and return job id
    """
    res = subprocess.run(
        [launcher, job_script],
        check=True,
        capture_output=True,
        text=True,
    )
    res = re.findall(r"Submitted batch job (.*)$", res.stdout).pop()
    return res.strip()


class HeaderReplace:
    def __init__(self, header_path: str):
        header_files = {
            "<GPU1_CONFIG>": "gpu1_config.slurm",
            "<GPU4_CONFIG>": "gpu4_config.slurm",
            "<CPU_CONFIG>": "cpu_config.slurm",
            "<ENV_LOAD>": "env_load.sh",
        }
        self.header_addenda = {}
        for k, h in header_files.items():
            fname = os.path.join(header_path, h)
            if not os.path.exists(fname):
                fname = full_path(h)

            with open(fname, "r") as fhandle:
                file_content = fhandle.read()

            self.header_addenda[k] = file_content

    def __call__(self, script: str):
        with open(script, "r") as fhandle:
            script_content = fhandle.read()

        for key, value in self.header_addenda.items():
            script_content = script_content.replace(key, value)

        with open(script, "w") as fhandle:
            fhandle.write(script_content)


def create_data_gen_commands(args: Arguments) -> GenDataResults:
    """
    Create data generation commands
    """
    with open(full_path("gen_data.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "JOBNAME": args.name_prefix,
        "<SCRIPTPATH>": _THISDIR,
        "<WORKDIR>": args.workdir,
        "<TSV>": args.tsv,
        "<REF>": args.ref,
        "<Timestamp>": args.timestamp,
        "<LAST_DATE>": args.last_date,
        "<N_COMP>": args.max_sequence_comparisons,
    }

    if args.first_date:
        replace_dict["\"<FIRST_DATE>\""] = args.first_date

    if args.fasta:
        replace_dict["\"<FASTA>\""] = args.fasta

    command_string = replace_helper(command_string, replace_dict)

    with open(os.path.join(args.workdir, "gen_cmd.sh"), "w") as whandle:
        whandle.write(command_string)
        fname = whandle.name

    # Fix header
    args.header_fixer(fname)

    if args.launch:
        jobid = launch_job(fname)
    else:
        jobid = "JOBNOTLAUNCHED"

    return GenDataResults(
        script=fname,
        jobid=jobid,
        generator_train_fa=os.path.join(args.workdir, f"decoder_data_unenumerated_{args.timestamp}.train.fa"),
        generator_val_fa=os.path.join(args.workdir, f"decoder_data_unenumerated_{args.timestamp}.val.fa"),
        competition_train_json=os.path.join(
            args.workdir,
            f"precomputed_competition_pairings_{args.last_date}_AsiaEurope_leading_weeks_{args.timestamp}.train.json"
        )
    )


def create_finetune_commands(args: Arguments, gen_results: GenDataResults) -> FinetuneResults:
    """
    Create finetuning command
    """
    with open(full_path("finetune.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "<WORKDIR>": args.workdir,
        "<SCRIPTPATH>": _THISDIR,
        "<TRAIN>": gen_results.generator_train_fa,
        "<VAL>": gen_results.generator_val_fa,
        "<PRETRAINED>": args.pretrained_ckpt,
        "<Timestamp>": args.timestamp,
        "JOBNAME": args.name_prefix,
        "<GENDATA>": gen_results.jobid,
        "<FINETUNE_BATCH_SIZE>": args.finetune_batch_size,
    }

    command_string = replace_helper(command_string, replace_dict)

    shutil.copy(full_path("load_fa.py"), args.workdir)    
    shutil.copy(full_path("calc_epoch_count.py"), args.workdir)

    with open(os.path.join(args.workdir, "finetune.sh"), "w") as fhandle:
        fhandle.write(command_string)
        script = fhandle.name

    # Fix header
    args.header_fixer(script)

    if args.launch:
        jobid = launch_job(script)
    else:
        jobid = "JOBNOTLAUNCHED"

    return FinetuneResults(
        script=script,
        jobid=jobid,
        finetune_model_path=os.path.join(args.workdir, "models", f"sda_{args.name_prefix}_{args.timestamp}")
    )


def create_competition_commands(
    args: Arguments,
    finetune_results: FinetuneResults,
    gen_results: GenDataResults,
) -> CompetitionRawResults:
    """
    Create competition command
    """
    with open(full_path("train_competition.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "JOBNAME": args.name_prefix,
        "<SCRIPTPATH>": _THISDIR,
        "<WORKDIR>": args.workdir,
        "<TRAIN>": gen_results.competition_train_json,
        "<REF>": args.ref,
        "<SDA>": finetune_results.finetune_model_path,
        "<Timestamp>": args.timestamp,
        "<FINETUNE>": finetune_results.jobid,
        "<COMPETITION_BATCH_SIZE>": args.competition_batch_size,
    }

    command_string = replace_helper(command_string, replace_dict)

    with open(os.path.join(args.workdir, "train_competition.sh"), "w") as fhandle:
        fhandle.write(command_string)
        fname = fhandle.name

    # Fix header
    args.header_fixer(fname)

    if args.launch:
        jobid = launch_job(fname)
    else:
        jobid = "JOBNOTLAUNCHED"

    return CompetitionRawResults(
        script=fname,
        jobid=jobid,
        checkpoint_storage=os.path.join(args.workdir, "models", f"competition_{args.name_prefix}_{args.timestamp}")
    )


def benchmark_competition_checkpoints(args: Arguments, competition_results: CompetitionRawResults, finetune_results: FinetuneResults):
    """
    Benchmark Competition results
    """
    with open(full_path("validate_checkpoints.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "JOBNAME": args.name_prefix,
        "<SCRIPTPATH>": _THISDIR,
        "<WORKDIR>": args.workdir,
        "<TRAINDIR>": competition_results.checkpoint_storage,
        "<SDA>": finetune_results.finetune_model_path,
        "<REF>": args.ref,
        "<TSV>": args.tsv,
        "<Timestamp>": args.timestamp,
        "<LAST_DATE>": args.last_date,
        "<COMPETITION>": competition_results.jobid,
    }

    command_string = replace_helper(command_string, replace_dict)

    with open(os.path.join(args.workdir, "validate_checkpoints.sh"), "w") as fhandle:
        fhandle.write(command_string)
        script = fhandle.name

    # Fix header
    args.header_fixer(script)

    if args.launch:
        jobid = launch_job(script)
    else:
        jobid = "JOBNOTLAUNCHED"

    log_prefix = os.path.join(args.workdir, f"{args.name_prefix}_competition_validation.{jobid}")

    return ValidationResults(
        script=script,
        jobid=jobid,
        validation_log_prefix=log_prefix,
    )


def quark_init_command(args: Arguments, finetune_results: FinetuneResults) -> QuarkInitResults:
    """
    Launch quark init to create initial data pool
    """
    with open(full_path("quark_init.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "JOBNAME": args.name_prefix,
        "<SCRIPTPATH>": _THISDIR,
        "<FINETUNE>": finetune_results.jobid,
        "<WORKDIR>": args.workdir,
        "<Timestamp>": args.timestamp,
        "<SDA>": finetune_results.finetune_model_path,
    }

    command_string = replace_helper(command_string, replace_dict)

    with open(os.path.join(args.workdir, "quark_init.sh"), "w") as fhandle:
        fhandle.write(command_string)
        script = fhandle.name

    # Fix header
    args.header_fixer(script)

    if args.launch:
        jobid = launch_job(script)
    else:
        jobid = "JOBNOTLAUNCHED"

    return QuarkInitResults(
        script=script,
        jobid=jobid,
        init_file=os.path.join(args.workdir, f"quark_init_{args.timestamp}.json")
    )


def quark_finetune_command(args: Arguments, gen_results: GenDataResults, finetune_results: FinetuneResults, quark_init: QuarkInitResults, validation_results: ValidationResults):
    """
    Launch quark finetuning step
    """
    with open(full_path("quark_finetune.sh"), "r") as fhandle:
        command_string = fhandle.read()

    replace_dict = {
        "JOBNAME": args.name_prefix,
        "<SCRIPTPATH>": _THISDIR,
        "<COMPETITION_VALIDATION>": validation_results.jobid,
        "<QUARK_INIT>": quark_init.jobid,
        "<WORKDIR>": args.workdir,
        "<INIT_SEQUENCES>": quark_init.init_file,
        "<LOG_PREFIX>": validation_results.validation_log_prefix,
        "<TRAIN_SEQUENCES>": gen_results.generator_train_fa,
        "<Timestamp>": args.timestamp,
        "<SDA>": finetune_results.finetune_model_path,
        "<QUARK_GEN_BATCH_SIZE>": args.quark_gen_batch_size,
        "<QUARK_TRAIN_BATCH_SIZE>": args.quark_train_batch_size,
    }

    if args.quark_gradient_checkpoint:
        replace_dict["<GRADIENT_CHECKPOINT_ARGS>"] = "--gradient_checkpointing"
    else:
        replace_dict["<GRADIENT_CHECKPOINT_ARGS>"] = '""'

    command_string = replace_helper(command_string, replace_dict)

    with open(os.path.join(args.workdir, "quark_finetune.sh"), "w") as fhandle:
        fhandle.write(command_string)
        script = fhandle.name

    shutil.copy(full_path("best_competition_ckpt.py"), args.workdir)    
    shutil.copy(full_path("quantile_spec.json"), args.workdir)

    # Fix header
    args.header_fixer(script)

    if args.launch:
        jobid = launch_job(script)
    else:
        jobid = "JOBNOTLAUNCHED"

    return QuarkFinetuneResults(
        script=script,
        jobid=jobid,
        model_path=os.path.join(args.workdir, "models", f"quark_{args.name_prefix}_{args.timestamp}")
    )


def main(args: Arguments):
    gen_data_results = create_data_gen_commands(args)
    finetune_results = create_finetune_commands(args, gen_data_results)
    competition_results = create_competition_commands(args, finetune_results, gen_data_results)
    benchmark_competition_results = benchmark_competition_checkpoints(args, competition_results, finetune_results)
    quark_init_results = quark_init_command(args, finetune_results)
    quark_finetune_results = quark_finetune_command(
        args,
        gen_data_results,
        finetune_results,
        quark_init_results,
        benchmark_competition_results,
    )

    with open(os.path.join(args.workdir, "job_board.json"), "w") as fhandle:
        json.dump(
            {
                "GenData": gen_data_results._asdict(),
                "Finetune": finetune_results._asdict(),
                "Competition": competition_results._asdict(),
                "CompetitionBenchmarking": benchmark_competition_results._asdict(),
                "QuarkInit": quark_init_results._asdict(),
                "QuarkFinetune": quark_finetune_results._asdict(),
            },
            fhandle,
            indent=4,
        )


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments, ), description="Run Quark Pipeline using SLURM")
    args_, = parser.parse_args_into_dataclasses()
    main(args_)
    