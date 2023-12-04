# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=4
#SBATCH --mem-per-cpu=1200
#SBATCH --gres=gpu:v100:4
#SBATCH --exclude=hal02