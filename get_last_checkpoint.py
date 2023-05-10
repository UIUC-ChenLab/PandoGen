# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
from transformers.trainer_utils import get_last_checkpoint
import sys


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <output directory>")
else:
    directory = sys.argv[1]
    print(get_last_checkpoint(directory))
