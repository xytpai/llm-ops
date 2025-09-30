import os
import sys
import json
import glob
import torch
import inspect
from tqdm import tqdm
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from utils import divide_and_check_no_remainder, benchmark_func, SDPARecord, SDPAAnalyzer
from bench_config import cfg


class AttentionInterface(ABC):
    def __init__(self, name='default', device='cuda'):
        self.name = name
        self.device = device

    def create_inputs(self, meta) -> None:
        self.batch_size = meta.batch_size
        self.seq_len = meta.seq_len
        self.nhead_q = meta.nhead_q
        self.nhead_kv = meta.nhead_kv
        self.head_dim = meta.head_dim
        if isinstance(meta.dtype, str):
            self.dtype = eval('torch.' + meta.dtype)
        else:
            self.dtype = meta.dtype
        self.q = torch.randn((self.batch_size, self.seq_len, self.nhead_q,
                             self.head_dim), dtype=self.dtype, device=self.device)
        self.k = torch.randn((self.batch_size, self.seq_len, self.nhead_kv,
                             self.head_dim), dtype=self.dtype, device=self.device)
        self.v = torch.randn((self.batch_size, self.seq_len, self.nhead_kv,
                             self.head_dim), dtype=self.dtype, device=self.device)

    @abstractmethod
    def eval(self):
        pass

    @benchmark_func()
    def _eval(self) -> float:
        return self.eval()

    def __call__(self, meta):
        self.create_inputs(meta)
        device_us = self._eval()
        record = SDPARecord(
            batch_size=self.batch_size,
            num_heads_q=self.nhead_q,
            num_heads_kv=self.nhead_kv,
            head_dim_qk=self.head_dim,
            head_dim_v=self.head_dim,
            seq_len_q=self.seq_len,
            seq_len_kv=self.seq_len,
            dtype=str(self.dtype),
            is_causal=True)
        ana = SDPAAnalyzer(record)
        total_flop = ana.getFLOP()
        tflops = total_flop * 1e-6 / device_us
        meta.tflops[self.name] = tflops
        return meta


@dataclass
class AttentionOpMeta:
    model_name: str = ""
    dtype: str = ""
    batch_size: int = 0
    seq_len: int = 0
    nhead_q: int = 0
    nhead_kv: int = 0
    head_dim: int = 0
    flag: str = ""
    count: int = 1
    is_causal: bool = True
    tp_size: int = 1
    tflops: dict = field(default_factory=dict)


class ExtractAttentionsBase:
    def __init__(self, config_file):
        self.load_config(config_file)

    def set_dtype(self, dtype: str):
        self.dtype = dtype

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.config = config
        self.dtype = config['torch_dtype']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        if self.config.get('head_dim', None):
            self.head_dim = self.config['head_dim']
        else:
            self.head_dim = divide_and_check_no_remainder(
                self.hidden_size, self.config['num_attention_heads'])
        self.model_name = os.path.basename(config_file).replace('.json', '')

    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        nhead_q = divide_and_check_no_remainder(
            self.config['num_attention_heads'], tp_size)
        nhead_kv = divide_and_check_no_remainder(
            self.config['num_key_value_heads'], tp_size)
        head_dim = self.head_dim
        metas = [
            AttentionOpMeta(batch_size=batch_size, seq_len=seq_len, nhead_q=nhead_q, nhead_kv=nhead_kv,
                            head_dim=head_dim, dtype=self.dtype, flag="sdpa", count=self.num_hidden_layers,
                            model_name=self.model_name, tp_size=tp_size),
        ]
        return metas

    def metas(self, batch_size=1, seq_len=1, tp_size=1):
        metas = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_'):
                metas += method(batch_size=batch_size,
                                seq_len=seq_len, tp_size=tp_size)
        return metas

    @staticmethod
    def get_shapes(metas):
        args = []
        for meta in metas:
            args.append([meta.batch_size, meta.seq_len,
                        meta.nhead_q, meta.nhead_kv, meta.head_dim])
        args = sorted(set(map(tuple, args)))
        shapes = []
        for arg in args:
            batch_size = arg[0]
            seq_len = arg[1]
            nhead_q = arg[2]
            nhead_kv = arg[3]
            head_dim = arg[4]
            shapes.append([batch_size, seq_len, nhead_q, nhead_kv, head_dim])
        return shapes

    @staticmethod
    def benchmark(metas, interface: AttentionInterface):
        new_metas = []
        pbar = tqdm(total=len(metas))
        for meta in metas:
            try:
                new_meta = interface(meta)
                new_metas.append(new_meta)
            except Exception as e:
                print(f"Error in {meta}: {e}")
            pbar.update(1)
        return new_metas


class ExtractAttentionsDeepSeek(ExtractAttentionsBase):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.qk_nope_head_dim = self.config['qk_nope_head_dim']
        self.qk_rope_head_dim = self.config['qk_rope_head_dim']
        self.mha = True

    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        if self.mha:
            nhead_q = divide_and_check_no_remainder(
                self.config['num_attention_heads'], tp_size)
            nhead_kv = divide_and_check_no_remainder(
                self.config['num_key_value_heads'], tp_size)
            head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            metas = [
                AttentionOpMeta(batch_size=batch_size, seq_len=seq_len, nhead_q=nhead_q, nhead_kv=nhead_kv,
                                head_dim=head_dim, dtype=self.dtype, flag="sdpa", count=self.num_hidden_layers,
                                model_name=self.model_name, tp_size=tp_size),
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
    shapes = ExtractAttentionsBase.get_shapes(metas)
    for shape in shapes:
        batch_size, seq_len, nhead_q, nhead_kv, head_dim = shape
        print(f"{batch_size}, {seq_len}, {nhead_q}, {nhead_kv}, {head_dim}")

    class TorchAttentionTest(AttentionInterface):
        def __init__(self):
            super().__init__('torch', 'cuda')

        def eval(self):
            return F.scaled_dot_product_attention(self.q, self.k, self.v, dropout_p=0.0, is_causal=True)

    metas = ExtractAttentionsBase.benchmark(metas, TorchAttentionTest())
    for meta in metas:
        print(meta)
