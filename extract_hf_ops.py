import os
import re
import sys
import glob
import json
import inspect


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class ExtractHFOpsBase:
    def __init__(self, config_file):
        self.load_config(config_file)
    
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

    def extract_attention_gemms(self, m=1, tp_size=1):
        q_n = self.head_dim * self.config['num_attention_heads']
        q_n = divide_and_check_no_remainder(q_n, tp_size)
        kv_n = self.head_dim * self.config['num_key_value_heads']
        kv_n = divide_and_check_no_remainder(kv_n, tp_size)
        qkv_k = self.hidden_size
        metas = [
            f"m={m}, n={q_n + 2 * kv_n}, k={qkv_k}, dtype={self.dtype}, flag=qkv_proj",
            f"m={m}, n={qkv_k}, k={q_n}, dtype={self.dtype}, flag=o_proj",
        ]
        return metas, self.num_hidden_layers
    
    def extract_ffn_dense_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_dense_layers > 0:
            gate_up_k = self.hidden_size
            gate_up_n = divide_and_check_no_remainder(self.intermediate_size, tp_size)
            down_k = gate_up_n
            down_n = gate_up_k
            metas = [
                f"m={m}, n={gate_up_n * 2}, k={gate_up_k}, dtype={self.dtype}, flag=gate_up_proj", 
                f"m={m}, n={down_n}, k={down_k}, dtype={self.dtype}, flag=down_proj", 
            ]
        return metas, self.ffn_dense_layers

    def extract_ffn_moe_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_moe_layers > 0:
            metas = [
                f"m={m}, n={self.num_experts}, k={self.hidden_size}, dtype={self.dtype}, flag=experts_selection",
            ]
        return metas, self.ffn_moe_layers
    
    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        nhead_q = divide_and_check_no_remainder(self.config['num_attention_heads'], tp_size)
        nhead_kv = divide_and_check_no_remainder(self.config['num_key_value_heads'], tp_size)
        head_dim = self.head_dim
        metas = [
            f"batch_size={batch_size}, seq_len={seq_len}, nhead_q={nhead_q}, nhead_kv={nhead_kv}, head_dim={head_dim}, dtype={self.dtype}", 
        ]
        return metas, self.num_hidden_layers
        
    def check_num_parameters(self):
        gemms = []
        num_params = 0
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_') and name.endswith('_gemms'):
                method_gemms, method_gemms_cnt = method(1, 1)
                gemms += method_gemms * method_gemms_cnt
        for gemm_meta in gemms:
            m, n, k, _ = nums = re.findall(r"\d+", gemm_meta)
            num_params += int(n) * int(k)
        num_params += self.hidden_size * self.config['vocab_size'] # embedding
        if self.ffn_moe_layers > 0:
            num_params += 3 * self.hidden_size * self.moe_intermediate_size * self.num_experts * self.ffn_moe_layers
        num_params = num_params / 1000 / 1000 / 1000
        expected_params = self.config.get('parameters', 1)
        relative_diff = abs(num_params - expected_params) / expected_params
        assert relative_diff < 1e-2, "The calculated number of parameters {}B is different from the expected {}B".format(num_params, expected_params)
        return num_params
    
    def extract_gemm_shapes(self, m=1, tp_size=1):
        gemms = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('extract_') and name.endswith('_gemms'):
                method_gemms, _ = method(m, tp_size)
                gemms += method_gemms
        shapes = []
        for gemm_meta in gemms:
            # print(gemm_meta)
            m, n, k, _ = nums = re.findall(r"\d+", gemm_meta)
            shapes.append(f"{m}, {n}, {k}")
        shapes = sorted(list(set(shapes)))
        return shapes


class ExtractHFOpsDeepSeek(ExtractHFOpsBase):
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
            q_b_n = divide_and_check_no_remainder(self.num_heads * self.qk_head_dim, tp_size)
            kv_a_k = self.hidden_size
            kv_a_n = self.kv_lora_rank + self.qk_rope_head_dim
            kv_b_k = self.kv_lora_rank
            kv_b_n = divide_and_check_no_remainder(self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), tp_size)
            o_k = divide_and_check_no_remainder(self.num_heads * self.v_head_dim, tp_size)
            o_n = self.hidden_size
            metas = [
                f"m={m}, n={q_a_n}, k={q_a_k}, dtype={self.dtype}, flag=q_a_proj",
                f"m={m}, n={q_b_n}, k={q_b_k}, dtype={self.dtype}, flag=q_b_proj",
                f"m={m}, n={kv_a_n}, k={kv_a_k}, dtype={self.dtype}, flag=kv_a_proj",
                f"m={m}, n={kv_b_n}, k={kv_b_k}, dtype={self.dtype}, flag=kv_b_proj",
                f"m={m}, n={o_n}, k={o_k}, dtype={self.dtype}, flag=o_proj",
            ]
        else:
            qkv_a_k = self.hidden_size
            qkv_a_n = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            q_b_k = self.q_lora_rank
            q_b_n = divide_and_check_no_remainder(self.num_heads * self.qk_head_dim, tp_size)
            kv_b_k = self.kv_lora_rank
            kv_b_n = divide_and_check_no_remainder(self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), tp_size)
            o_k = divide_and_check_no_remainder(self.num_heads * self.v_head_dim, tp_size)
            o_n = self.hidden_size
            metas = [
                f"m={m}, n={qkv_a_n}, k={qkv_a_k}, dtype={self.dtype}, flag=qkv_a_proj",
                f"m={m}, n={q_b_n}, k={q_b_k}, dtype={self.dtype}, flag=q_b_proj",
                f"m={m}, n={kv_b_n}, k={kv_b_k}, dtype={self.dtype}, flag=kv_b_proj",
                f"m={m}, n={o_n}, k={o_k}, dtype={self.dtype}, flag=o_proj",
            ]
        return metas, self.num_hidden_layers
    
    def extract_ffn_shared_gemms(self, m=1, tp_size=1):
        metas = []
        if self.ffn_moe_layers > 0:
            gate_up_k = self.hidden_size
            gate_up_n = divide_and_check_no_remainder(self.moe_intermediate_size * self.config['n_shared_experts'], tp_size)
            down_k = gate_up_n
            down_n = gate_up_k
            metas = [
                f"m={m}, n={gate_up_n * 2}, k={gate_up_k}, dtype={self.dtype}, flag=gate_up_proj", 
                f"m={m}, n={down_n}, k={down_k}, dtype={self.dtype}, flag=down_proj", 
            ]
        return metas, self.ffn_moe_layers
        
    def extract_attentions(self, batch_size=1, seq_len=1, tp_size=1):
        if self.mha:
            nhead_q = divide_and_check_no_remainder(self.config['num_attention_heads'], tp_size)
            nhead_kv = divide_and_check_no_remainder(self.config['num_key_value_heads'], tp_size)
            head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            metas = [
                f"batch_size={batch_size}, seq_len={seq_len}, nhead_q={nhead_q}, nhead_kv={nhead_kv}, head_dim={head_dim}, dtype={self.dtype}", 
            ]
            return metas, self.num_hidden_layers
        else:
            raise NotImplementedError("Not implemented for non-MHA.")


def get_extract_method(config_file):
    if 'DeepSeek-V3' in config_file:
        return ExtractHFOpsDeepSeek
    else:
        return ExtractHFOpsBase


if __name__ == '__main__':
    config_file = sys.argv[1]
    print(config_file)
    eo = get_extract_method(config_file)(config_file)
    gemm_shapes = eo.extract_gemm_shapes()
    for mnk in gemm_shapes:
        print(mnk)
    print("nparams:", eo.check_num_parameters(), "B")
