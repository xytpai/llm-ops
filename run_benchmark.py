import os
import sys
import glob
import csv
import torch
import torch.nn.functional as F
import csv
from dataclasses import asdict

import extract_gemms
import extract_attentions
from bench_config import cfg, fp_8_dtype


class TorchGemmTest(extract_gemms.GemmInterface):
    def __init__(self):
        super().__init__('torch', 'cuda')

    def eval(self):
        if self.x.dtype == fp_8_dtype:
            out = torch._scaled_mm(self.x, self.w.t(),
                    out_dtype=torch.bfloat16,
                    scale_a=self.scale_a,
                    scale_b=self.scale_b,
                    bias=None,
                )
        else:
            out = F.linear(self.x, self.w)
        return out


class TorchAttentionTest(extract_attentions.AttentionInterface):
    def __init__(self):
        super().__init__('torch', 'cuda')
    
    def eval(self):
        return F.scaled_dot_product_attention(self.q, self.k, self.v, dropout_p=0.0, is_causal=True)


def test_model_gemm(config_file):
    method_class = extract_gemms.get_method(config_file)
    method = method_class(config_file)
    print(config_file, "nparams:", method.check_num_parameters(), "B")
    cfg_ = cfg.gemm
    results = []
    for dtype in cfg_.dtypes:
        method.set_dtype(dtype)
        for tp_size in cfg_.tp_sizes:
            for m in cfg_.ms:
                metas = method.metas(m=m, tp_size=tp_size)
                metas = method_class.benchmark(metas, TorchGemmTest())
                results += metas
    return results


def test_model_sdpa(config_file):
    method_class = extract_attentions.get_method(config_file)
    method = method_class(config_file)
    cfg_ = cfg.attention
    results = []
    for dtype in cfg_.dtypes:
        method.set_dtype(dtype)
        for tp_size in cfg_.tp_sizes:
            for batch_size in cfg_.batch_sizes:
                for seq_len in cfg_.seq_lens:
                    metas = method.metas(batch_size=batch_size, seq_len=seq_len, tp_size=tp_size)
                    metas = method_class.benchmark(metas, TorchAttentionTest())
                    results += metas
    return results


def main():
    for op in cfg.test_scope:

        print(f"\ntest {op} ...\n")
        
        if op == 'gemm':
            results = []
            for file in cfg.files:
                results += test_model_gemm(file)
            results = sorted(results, key=lambda x: str(x))
            with open("out_gemm.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for u in results:
                    writer.writerow(asdict(u))
            
        elif op == 'attention':
            results = []
            for file in cfg.files:
                results += test_model_sdpa(file)
            results = sorted(results, key=lambda x: str(x))
            with open("out_sdpa.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for u in results:
                    writer.writerow(asdict(u))


if __name__ == "__main__":
    main()
