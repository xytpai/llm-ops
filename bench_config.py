import torch
from addict import Dict


cfg = Dict()
fp_8_dtype = torch.float8_e4m3fn


# Gemms
cfg.gemm.dtypes = [fp_8_dtype, torch.bfloat16]
cfg.gemm.tp_sizes = [1, 4, 8]
cfg.gemm.ms = [i * 1024 for i in [1, 2, 4]]


# Attention
cfg.attention.dtypes = [torch.bfloat16]
cfg.attention.tp_sizes = [1, 4]
cfg.attention.batch_sizes = [1, 4, 8]
cfg.attention.seq_lens = [1024, 2048]
