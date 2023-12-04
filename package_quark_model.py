# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from predict_decoder import *
import sys
from quark_finetune import QuarkModel
import shutil
import os

def load_quark_model_re(pth: str) -> QuarkModel:
    filename = os.path.join(pth, TRAINING_ARGS_NAME)
    training_args = torch.load(filename)
    model_args = training_args.model_args
    quark_model = make_model(model_args)
    weights = torch.load(os.path.join(pth, WEIGHTS_NAME), map_location="cpu")
    quark_model.load_state_dict(weights)
    return quark_model

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <quark training checkpoint> <output path>")
    sys.exit()

quark_model = load_quark_model_re(sys.argv[1])
quark_model.train_model.save_pretrained(sys.argv[2])
shutil.copy(os.path.join(sys.argv[1], TRAINING_ARGS_NAME), os.path.join(sys.argv[2], TRAINING_ARGS_NAME))
