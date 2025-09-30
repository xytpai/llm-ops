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
from utils import divide_and_check_no_remainder, benchmark_func
from bench_config import cfg


class GemmInterface(ABC):
    def __init__(self, name='default', device='cuda'):
        self.name = name
        self.device = device

    def create_inputs(self, meta) -> None:
        self.m = meta.m
        self.n = meta.m
        self.k = meta.k
        if isinstance(meta.dtype, str):
            self.dtype = eval('torch.' + meta.dtype)
        else:
            self.dtype = meta.dtype
        self.x = torch.randn((self.m, self.k), dtype=torch.float,
                             device=self.device).to(self.dtype)
        self.w = torch.randn((self.n, self.k), dtype=torch.float,
                             device=self.device).to(self.dtype)
        self.scale_a = torch.ones(1, dtype=torch.float, device=self.device)
        self.scale_b = torch.ones(1, dtype=torch.float, device=self.device)

    @abstractmethod
    def eval(self):
        pass

    @benchmark_func()
    def _eval(self) -> float:
        return self.eval()

    def __call__(self, meta):
        self.create_inputs(meta)
        device_us = self._eval()
        tflops = 2 * self.m * self.n * self.k / (device_us) / 1e6
        meta.tflops[self.name] = tflops
        return meta


@dataclass
class GEMMOpMeta:
    model_name: str = ""
    dtype: str = ""
    m: int = 0
    n: int = 0
    k: int = 0
    flag: str = ""
    count: int = 1
    tp_size: int = 1
    tflops: dict = field(default_factory=dict)


class ExtractGeemsBase:
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
        self.intermediate_size = config['intermediate_size']
        self.moe_intermediate_size = config.get('moe_intermediate_size', None)
        if self.config.get('head_dim', None):
            self.head_dim = self.config['head_dim']
        else:
            self.head_dim = divide_and_check_no_remainder(
                self.hidden_size, self.config['num_attention_heads'])

        if self.moe_intermediate_size is not None:
            self.ffn_dense_layers = 0
            self.ffn_moe_layers = self.num_hidden_layers
        else:
            self.ffn_dense_layers = self.num_hidden_layers
            self.ffn_moe_layers = 0
        self.num_experts = self.config.get('num_experts', None)
        self.model_name = os.path.basename(config_file).replace('.json', '')

    def extract_attention_gemms(self, m=1, tp_size=1):
        q_n = self.head_dim * self.config['num_attention_heads']
        q_n = divide_and_check_no_remainder(q_n, tp_size)
        kv_n = self.head_dim * self.config['num_key_value_heads']
        kv_n = divide_and_check_no_remainder(kv_n, tp_size)
        qkv_k = self.hidden_size
        metas = [
            GEMMOpMeta(
                m=m, n=(q_n + 2 * kv_n), k=qkv_k,
                dtype=self.dtype, flag="qkv_proj",
                count=self.num_hidden_layers,
                model_name=self.model_name,
                tp_size=tp_size),
            GEMMOpMeta(m=m, n=qkv_k, k=q_n,
                       dtype=self.dtype, flag="o_proj",
                       count=self.num_hidden_layers,
                       model_name=self.model_name,
                       tp_size=tp_size),
        ]
        return metas

    def extract_ffn_dense_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_dense_layers > 0:
            gate_up_k = self.hidden_size
            gate_up_n = divide_and_check_no_remainder(
                self.intermediate_size, tp_size)
            down_k = gate_up_n
            down_n = gate_up_k
            metas = [
                GEMMOpMeta(m=m, n=(gate_up_n * 2), k=gate_up_k,
                           dtype=self.dtype, flag="gate_up_proj",
                           count=self.ffn_dense_layers,
                           model_name=self.model_name,
                           tp_size=tp_size),
                GEMMOpMeta(m=m, n=down_n, k=down_k,
                           dtype=self.dtype, flag="down_proj",
                           count=self.ffn_dense_layers,
                           model_name=self.model_name,
                           tp_size=tp_size),
            ]
        return metas

    def extract_ffn_moe_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_moe_layers > 0:
            metas = [
                GEMMOpMeta(m=m, n=(self.num_experts), k=self.hidden_size,
                           dtype=self.dtype, flag="experts_selection",
                           count=self.ffn_moe_layers,
                           model_name=self.model_name,
                           tp_size=tp_size),
            ]
        return metas

    def check_num_parameters(self):
        metas = []
        num_params = 0
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_'):
                metas += method(m=1, tp_size=1)
        for meta in metas:
            num_params += meta.n * meta.k * meta.count
        num_params += self.hidden_size * self.config['vocab_size']  # embedding
        if self.ffn_moe_layers > 0:
            num_params += 3 * self.hidden_size * self.moe_intermediate_size * \
                self.num_experts * self.ffn_moe_layers
        num_params = num_params / 1000 / 1000 / 1000
        expected_params = self.config.get('parameters', 1)
        relative_diff = abs(num_params - expected_params) / expected_params
        assert relative_diff < 1e-2, "The calculated number of parameters {}B is different from the expected {}B".format(
            num_params, expected_params)
        return num_params

    def metas(self, m=1, tp_size=1):
        metas = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_'):
                metas += method(m=m, tp_size=tp_size)
        return metas

    @staticmethod
    def get_shapes(metas):
        mnks = []
        for meta in metas:
            mnks.append([meta.m, meta.n, meta.k])
        mnks = sorted(set(map(tuple, mnks)))
        return mnks

    @staticmethod
    def benchmark(metas, interface: GemmInterface):
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


class ExtractGeemsDeepSeek(ExtractGeemsBase):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.qk_nope_head_dim = self.config['qk_nope_head_dim']
        self.qk_rope_head_dim = self.config['qk_rope_head_dim']
        self.v_head_dim = self.config['v_head_dim']
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_lora_rank = self.config['q_lora_rank']
        self.kv_lora_rank = self.config['kv_lora_rank']
        self.num_heads = self.config['num_attention_heads']
        self.first_k_dense_replace = self.config['first_k_dense_replace']
        self.ffn_dense_layers = self.first_k_dense_replace
        self.ffn_moe_layers = self.num_hidden_layers - self.first_k_dense_replace
        self.num_experts = self.config['n_routed_experts']
        self.mha = True

    def extract_attention_gemms(self, m=1, tp_size=1):
        if self.mha:
            q_a_k = self.hidden_size
            q_a_n = self.q_lora_rank
            q_b_k = self.q_lora_rank
            q_b_n = divide_and_check_no_remainder(
                self.num_heads * self.qk_head_dim, tp_size)
            kv_a_k = self.hidden_size
            kv_a_n = self.kv_lora_rank + self.qk_rope_head_dim
            kv_b_k = self.kv_lora_rank
            kv_b_n = divide_and_check_no_remainder(
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), tp_size)
            o_k = divide_and_check_no_remainder(
                self.num_heads * self.v_head_dim, tp_size)
            o_n = self.hidden_size
            metas = [
                GEMMOpMeta(m=m, n=q_a_n, k=q_a_k, dtype=self.dtype, flag="q_a_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=q_b_n, k=q_b_k, dtype=self.dtype, flag="q_b_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=kv_a_n, k=kv_a_k, dtype=self.dtype, flag="kv_a_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=kv_b_n, k=kv_b_k, dtype=self.dtype, flag="kv_b_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=o_n, k=o_k, dtype=self.dtype, flag="o_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
            ]
        else:
            qkv_a_k = self.hidden_size
            qkv_a_n = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            q_b_k = self.q_lora_rank
            q_b_n = divide_and_check_no_remainder(
                self.num_heads * self.qk_head_dim, tp_size)
            kv_b_k = self.kv_lora_rank
            kv_b_n = divide_and_check_no_remainder(
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), tp_size)
            o_k = divide_and_check_no_remainder(
                self.num_heads * self.v_head_dim, tp_size)
            o_n = self.hidden_size
            metas = [
                GEMMOpMeta(m=m, n=qkv_a_n, k=qkv_a_k, dtype=self.dtype, flag="qkv_a_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=q_b_n, k=q_b_k, dtype=self.dtype, flag="q_b_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=kv_b_n, k=kv_b_k, dtype=self.dtype, flag="kv_b_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=o_n, k=o_k, dtype=self.dtype, flag="o_proj",
                           count=self.num_hidden_layers, model_name=self.model_name, tp_size=tp_size),
            ]
        return metas

    def extract_ffn_shared_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_moe_layers > 0:
            gate_up_k = self.hidden_size
            gate_up_n = divide_and_check_no_remainder(
                self.moe_intermediate_size * self.config['n_shared_experts'], tp_size)
            down_k = gate_up_n
            down_n = gate_up_k
            metas = [
                GEMMOpMeta(m=m, n=(gate_up_n * 2), k=gate_up_k, dtype=self.dtype, flag="gate_up_proj",
                           count=self.ffn_moe_layers, model_name=self.model_name, tp_size=tp_size),
                GEMMOpMeta(m=m, n=down_n, k=down_k, dtype=self.dtype, flag="down_proj",
                           count=self.ffn_moe_layers, model_name=self.model_name, tp_size=tp_size),
            ]
        return metas


def get_method(config_file):
    if 'DeepSeek-V3' in config_file:
        return ExtractGeemsDeepSeek
    else:
        return ExtractGeemsBase


if __name__ == '__main__':
    config_dir = sys.argv[1]
    config_files = sorted(glob.glob(f"{config_dir}/*.json"))
    metas = []
    for config_file in config_files:
        print(config_file)
        extractor = get_method(config_file)(config_file)
        print("nparams:", extractor.check_num_parameters(), "B")
        # for dtype in cfg.gemm.dtypes:
        if True:
            for tp_size in cfg.gemm.tp_sizes:
                for m in cfg.gemm.ms:
                    metas += extractor.metas(m=m, tp_size=tp_size)
    for meta in metas:
        print(meta)

    mnks = ExtractGeemsBase.get_shapes(metas)
    for mnk in mnks:
        print(f"{mnk[0]}, {mnk[1]}, {mnk[2]}")

    class TorchGemmTest(GemmInterface):
        def __init__(self):
            super().__init__('torch', 'cuda')

        def eval(self):
            return F.linear(self.x, self.w)

    metas = ExtractGeemsBase.benchmark(metas, TorchGemmTest())
    for meta in metas:
        print(meta)
