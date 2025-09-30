import torch
from addict import Dict


cfg = Dict()
fp_8_dtype = torch.float8_e4m3fn


cfg.files = [
    'hf_configs/DeepSeek-V3.1.json',
    'hf_configs/Llama-3.1-405B-Instruct.json',
    'hf_configs/Llama-3.1-70B-Instruct.json',
    'hf_configs/Qwen3-235B-A22B.json',
    'hf_configs/Qwen3-32B.json',
]
cfg.test_scope = [
    'gemm',
    'attention',
]


# Gemms
cfg.gemm.dtypes = ['float8_e4m3fn', 'bfloat16']
cfg.gemm.tp_sizes = [1, 4, 8]
cfg.gemm.ms = [i * 1024 for i in [1, 2, 4]]


# Attention
cfg.attention.dtypes = ['bfloat16']
cfg.attention.tp_sizes = [1, 4]
cfg.attention.batch_sizes = [1, 4, 8]
cfg.attention.seq_lens = [1024, 2048, 8192]
