import sys
import json
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
    config_file = sys.argv[1]
    print(config_file)
    extractor = get_method(config_file)(config_file)
    metas = extractor.metas()
    for meta in metas:
        print(meta)
