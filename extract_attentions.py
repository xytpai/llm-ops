import sys
import json
import glob
import inspect
from dataclasses import dataclass
from utils import divide_and_check_no_remainder
from bench_config import cfg


@dataclass
class AttentionOpMeta:
    batch_size: int
    seq_len: int
    nhead_q: int
    nhead_kv: int
    head_dim: int
    dtype: str
    flag: str = ""
    count: int = 1
    is_causal: bool = True


class ExtractAttentionsBase:
    def __init__(self, config_file):
        self.load_config(config_file)
    
    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f) 
        self.config = config
        self.dtype = config['torch_dtype']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.dtype = config['torch_dtype']
        if self.config.get('head_dim', None):
            self.head_dim = self.config['head_dim']
        else:
            self.head_dim = divide_and_check_no_remainder(
                self.hidden_size, self.config['num_attention_heads'])
    
    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        nhead_q = divide_and_check_no_remainder(self.config['num_attention_heads'], tp_size)
        nhead_kv = divide_and_check_no_remainder(self.config['num_key_value_heads'], tp_size)
        head_dim = self.head_dim
        metas = [
            AttentionOpMeta(batch_size=batch_size, seq_len=seq_len, nhead_q=nhead_q, nhead_kv=nhead_kv, 
                head_dim=head_dim, dtype=self.dtype, flag="sdpa", count=self.num_hidden_layers),
        ]
        return metas

    def metas(self, batch_size=1, seq_len=1, tp_size=1):
        metas = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_'):
                metas += method(batch_size=batch_size, seq_len=seq_len, tp_size=tp_size)
        return metas


class ExtractAttentionsDeepSeek(ExtractAttentionsBase):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.qk_nope_head_dim = self.config['qk_nope_head_dim']
        self.qk_rope_head_dim = self.config['qk_rope_head_dim']
        self.mha = True
    
    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        if self.mha:
            nhead_q = divide_and_check_no_remainder(self.config['num_attention_heads'], tp_size)
            nhead_kv = divide_and_check_no_remainder(self.config['num_key_value_heads'], tp_size)
            head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            metas = [
                AttentionOpMeta(batch_size=batch_size, seq_len=seq_len, nhead_q=nhead_q, nhead_kv=nhead_kv, 
                    head_dim=head_dim, dtype=self.dtype, flag="sdpa", count=self.num_hidden_layers),
            ]
            return metas
        else:
            raise NotImplementedError("Not implemented for non-MHA.")


def get_method(config_file):
    if 'DeepSeek-V3' in config_file:
        return ExtractAttentionsDeepSeek
    else:
        return ExtractAttentionsBase


if __name__ == '__main__':
    config_dir = sys.argv[1]
    config_files = sorted(glob.glob(f"{config_dir}/*.json"))
    metas = []
    for config_file in config_files:
        print(config_file)
        extractor = get_method(config_file)(config_file)
        cfg_ = cfg.attention
        # for dtype in cfg_.dtypes:
        if True:
            for tp_size in cfg_.tp_sizes:
                for batch_size in cfg_.batch_sizes:
                    for seq_len in cfg_.seq_lens:
                        metas += extractor.metas(
                            batch_size=batch_size, seq_len=seq_len, tp_size=tp_size)
    for meta in metas:
        print(meta)
    args = []
    for meta in metas:
        args.append([meta.batch_size, meta.seq_len, meta.nhead_q, meta.nhead_kv, meta.head_dim])
    args = sorted(set(map(tuple, args)))
    from aiter_test_flash_attn import test_case
    from tqdm import tqdm
    pbar = tqdm(total=len(args))
    results = []
    for arg in args:
        batch_size = arg[0]
        seq_len = arg[1]
        nhead_q = arg[2]
        nhead_kv = arg[3]
        head_dim = arg[4]
        forward_tflops_bf16, backward_tflops_bf16 = test_case(batch_size, seq_len, nhead_q, nhead_kv, head_dim)
        # print(f"{batch_size}, {seq_len}, {nhead_q}, {nhead_kv}, {head_dim}")
        results.append([batch_size, seq_len, nhead_q, nhead_kv, head_dim, forward_tflops_bf16, backward_tflops_bf16])
        pbar.update(1)
    results = [['batch_size', 'seq_len', 'nhead_q', 'nhead_kv', 'head_dim', 'forward_tflops_bf16', 'backward_tflops_bf16']] + results
    import csv
    with open("output_attns.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)
