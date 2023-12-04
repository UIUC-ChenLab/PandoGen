# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
# 2023-04-29 01:17:49,990 INFO AUC (cutoff=50, checkpoint=$/home/aramach4/COVID19/UniProt_2022_Oct_13_04_00_17_CDT/UniRef_download_2022_Oct_13_04_00_17_CDT/uniref50/decoder_finetuning_delta_2023_Apr_28_02_11_25_CDT/models/competition_lr_1e_5_2023_Apr_28_10_14_03_CDT/checkpoint-2241) = 0.9172932330827068
# 2023-04-29 01:19:09,831 INFO AUC (cutoff=50, checkpoint=$/home/aramach4/COVID19/UniProt_2022_Oct_13_04_00_17_CDT/UniRef_download_2022_Oct_13_04_00_17_CDT/uniref50/decoder_finetuning_delta_2023_Apr_28_02_11_25_CDT/models/competition_lr_1e_5_2023_Apr_28_10_14_03_CDT/checkpoint-4482) = 0.9020130972592771
# 2023-04-29 01:20:29,964 INFO AUC (cutoff=50, checkpoint=$/home/aramach4/COVID19/UniProt_2022_Oct_13_04_00_17_CDT/UniRef_download_2022_Oct_13_04_00_17_CDT/uniref50/decoder_finetuning_delta_2023_Apr_28_02_11_25_CDT/models/competition_lr_1e_5_2023_Apr_28_10_14_03_CDT/checkpoint-6723) = 0.9102595197671599
# 2023-04-29 01:21:49,866 INFO AUC (cutoff=50, checkpoint=$/home/aramach4/COVID19/UniProt_2022_Oct_13_04_00_17_CDT/UniRef_download_2022_Oct_13_04_00_17_CDT/uniref50/decoder_finetuning_delta_2023_Apr_28_02_11_25_CDT/models/competition_lr_1e_5_2023_Apr_28_10_14_03_CDT/checkpoint-8964) = 0.8834586466165414

import sys
import re

fname = sys.argv[1]

with open(fname) as fhandle:
    lines = [x for x in fhandle if "INFO AUC" in x]

pattern = re.compile(r".*checkpoint=(.*)\) = (\S+)$")

items = []

for l in lines:
    res = pattern.match(l)
    if not res:
        raise ValueError(f"Cannot process line {l}")
    items.append(res.groups())

items.sort(key=lambda x: float(x[1]), reverse=True)
print(items[0][0])
