# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import sys
import math

total_iters = int(sys.argv[1])
n_examples = int(sys.argv[2])
batch_size = int(sys.argv[3])
n_acc = int(sys.argv[4])
n_gpus = int(sys.argv[5])

n_iters_per_epoch = n_examples / (batch_size * n_acc * n_gpus)
n_epochs = total_iters / n_iters_per_epoch

print(math.ceil(n_epochs))

