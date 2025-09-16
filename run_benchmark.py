import os
import sys
import glob
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import benchmark_func, SDPARecord, SDPAAnalyzer
import extract_gemms
import extract_attentions
from bench_config import cfg, fp_8_dtype


torch.set_default_device('cuda')


@benchmark_func()
def run_torch_gemm(x, w, scale_a, scale_b):
    if x.dtype == fp_8_dtype:
        out = torch._scaled_mm(x, w.t(),
                out_dtype=torch.bfloat16,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
            )
    else:
        out = F.linear(x, w)
    return out


def test_gemm(dtype, m, n, k):
    x = torch.randn((m, k), dtype=torch.float).to(dtype)
    w = torch.randn((n, k), dtype=torch.float).to(dtype)
    scale_a = torch.ones(1, dtype=torch.float)
    scale_b = torch.ones(1, dtype=torch.float)
    device_us = float(run_torch_gemm(x, w, scale_a, scale_b))
    tflops = 2 * m * n * k / (device_us) / 1e6
    return tflops


@benchmark_func()
def run_torch_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    return out


def test_sdpa(dtype, batch_size, seq_len, nhead_q, nhead_kv, head_dim):
    q = torch.randn((batch_size, seq_len, nhead_q, head_dim), dtype=dtype)
    k = torch.randn((batch_size, seq_len, nhead_kv, head_dim), dtype=dtype)
    v = torch.randn((batch_size, seq_len, nhead_kv, head_dim), dtype=dtype)
    device_us = float(run_torch_sdpa(q, k, v))
    record = SDPARecord(
        batch_size=batch_size,
        num_heads_q=nhead_q,
        num_heads_kv=nhead_kv,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        dtype=str(dtype),
        is_causal=True)
    ana = SDPAAnalyzer(record)
    total_flop = ana.getFLOP()
    tflops = total_flop * 1e-6 / device_us
    return tflops


def test_model_gemm(config_file):
    print(config_file)
    method = extract_gemms.get_method(config_file)(config_file)
    print("nparams:", method.check_num_parameters(), "B")
    cfg_ = cfg.gemm
    pbar = tqdm(total=len(cfg_.dtypes) * len(cfg_.tp_sizes) * len(cfg_.ms))
    results = []
    for dtype in cfg_.dtypes:
        for tp_size in cfg_.tp_sizes:
            for m in cfg_.ms:
                metas = method.metas(m=m, tp_size=tp_size)
                for meta in metas:
                    m = meta.m
                    n = meta.n
                    k = meta.k
                    tflops = test_gemm(dtype, m, n, k)
                    results.append([config_file, dtype, tp_size, m, n, k, tflops])
                pbar.update(1)
    return results


def test_model_sdpa(config_file):
    print(config_file)
    method = extract_attentions.get_method(config_file)(config_file)
    cfg_ = cfg.attention
    pbar = tqdm(total=len(cfg_.dtypes) * len(cfg_.tp_sizes) * len(cfg_.batch_sizes) * len(cfg_.seq_lens))
    results = []
    for dtype in cfg_.dtypes:
        for tp_size in cfg_.tp_sizes:
            for batch_size in cfg_.batch_sizes:
                for seq_len in cfg_.seq_lens:
                    metas = method.metas(batch_size=batch_size, seq_len=seq_len, tp_size=tp_size)
                    for meta in metas:
                        batch_size = meta.batch_size
                        seq_len = meta.seq_len
                        nhead_q = meta.nhead_q
                        nhead_kv = meta.nhead_kv
                        head_dim = meta.head_dim
                        tflops = test_sdpa(dtype, batch_size, seq_len, nhead_q, nhead_kv, head_dim)
                        results.append([config_file, dtype, batch_size, seq_len, nhead_q, nhead_kv, head_dim, tflops])
                    pbar.update(1)
    return results


if __name__ == "__main__":
    config_dir = sys.argv[1]
    config_files = sorted(glob.glob(f"{config_dir}/*.json"))

    print('\ntest sdpas ...\n')
    attention_results = []
    for config_file in config_files:
        attention_results += test_model_sdpa(config_file)
    datas = [['config_file', 'dtype', 'batch_size', 'seq_len', 'nhead_q', 'nhead_kv', 'head_dim', 'tflops']]
    datas += attention_results
    with open("output_attentions.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(datas)

    print('\ntest gemms ...\n')
    gemm_results = []
    for config_file in config_files:
        gemm_results += test_model_gemm(config_file)
    datas = [['config_file', 'dtype', 'tp_size', 'm', 'n', 'k', 'tflops']]
    datas += gemm_results
    with open("output_gemms.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(datas)
