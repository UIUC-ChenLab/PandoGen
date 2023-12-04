# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import pysam
import sys
import logging

logging.basicConfig(level=logging.INFO)

with pysam.FastaFile(sys.argv[1]) as fhandle:
    ref = fhandle.references
    logging.info(f"Total of {len(ref)} references")

