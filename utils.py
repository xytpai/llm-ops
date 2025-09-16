

import copy
import torch
import pandas as pd
from dataclasses import dataclass


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


@dataclass
class SDPARecord:
    batch_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim_qk: int
    head_dim_v: int
    seq_len_q: int
    seq_len_kv: int
    dtype: str
    is_causal: bool = True


class SDPAAnalyzer:
    def __init__(self, record: SDPARecord):
        self.batch_size = record.batch_size
        self.num_heads_q = record.num_heads_q
        self.num_heads_kv = record.num_heads_kv
        self.head_dim_qk = record.head_dim_qk
        self.head_dim_v = record.head_dim_v
        self.seq_len_q = record.seq_len_q
        self.seq_len_kv = record.seq_len_kv
        self.dtype = record.dtype
        self.is_causal = record.is_causal
        if record.dtype == "torch.bfloat16" or record.dtype == "torch.float16":
            self.element_bytes = 2
        elif record.dtype == "torch.float":
            self.element_bytes = 4
        else:
            raise "Data type not supported"
        assert self.num_heads_q % self.num_heads_kv == 0

    def getFLOP(self):
        L2 = self.seq_len_q * self.seq_len_kv
        if self.is_causal:
            n = min(self.seq_len_q, self.seq_len_kv)
            L2eff = (self.seq_len_kv + self.seq_len_kv - n + 1) * n / 2.0
        else:
            L2eff = L2
        BH = self.batch_size * self.num_heads_q
        qk_FLOP = 2 * BH * L2eff * self.head_dim_qk
        qksv_FLOP = 2 * BH * L2 * self.head_dim_v
        return qk_FLOP + qksv_FLOP

    def getFlashAttnBytes(self):
        read_q = self.batch_size * self.num_heads_q * self.seq_len_q * self.head_dim_qk
        read_k = (
            self.batch_size * self.num_heads_kv * self.seq_len_kv * self.head_dim_qk
        )
        read_v = self.batch_size * self.num_heads_kv * self.seq_len_kv * self.head_dim_v
        write_o = self.batch_size * self.num_heads_q * self.seq_len_q * self.head_dim_v
        return (read_q + read_k + read_v + write_o) * self.element_bytes


def benchmark_func(num_iters=101, num_warmup=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            device = torch.cuda.current_device()
            input_bytes = sum([arg.nbytes for arg in args if isinstance(arg, torch.Tensor) and arg.device.index == device])
            input_bytes += 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.max_memory_allocated(device)
            _ = func(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            after = torch.cuda.max_memory_allocated(device)
            new_allocated_bytes = after - before
            properties = torch.cuda.get_device_properties(device)
            free_memory = torch.cuda.mem_get_info(device)[0]
            cache_size = min(
                getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
                (free_memory - new_allocated_bytes) * 0.9,
            )
            cache_size = max(cache_size, 0)
            num = int((cache_size + input_bytes - 1) // input_bytes)
            num = min(num, num_iters)
            rotate_args = [
                (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num - 1)
            ] + [(args, kwargs)]
            for _ in range(num_warmup):
                _ = func(*args, **kwargs)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=False,
                with_stack=False,
                with_modules=True
            ) as prof:
                for idx in range(num_iters):
                    args, kwargs = rotate_args[idx % num]
                    data = func(*args, **kwargs)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000))
            time_us = get_trace_perf(prof, num_iters)
            # print(time_us)
            return time_us
        return wrapper
    return decorator


def post_process_data(df, num_iters, warm_iter=1):
    """remove abnormal data"""

    device_df = df[df["device_type"].astype(str).str.contains("DeviceType.CUDA")]
    # print("devicedf is ", device_df)
    if device_df.empty:
        return [], 0
    kernels_num = int(len(device_df) / num_iters)

    act_iters = num_iters
    valid_n = len(device_df)
    dropped_indexs = []
    if len(device_df) % num_iters == 0:
        kernels_num = int(len(device_df) / num_iters)
    else:
        ##get correct kernel num
        name_list = device_df["name"].tolist()
        max_kernel_num = 20
        n = len(name_list)
        for step in range(1, min(max_kernel_num, n // 2 + 1)):
            sub_list = [name_list[i] for i in range(step)]
            m = len(sub_list)

            valid_n = int(n / m) * m
            pattern_match = all(
                name_list[i] == sub_list[i % m] for i in range(int(n / m) * m)
            )
            if pattern_match:
                kernels_num = m
                act_iters = valid_n / m
                break
        dropped_indexs = device_df.iloc[valid_n:].index.tolist()
        if kernels_num == 0:
            print("data missed, the time may be inaccurate!")

    test_df = device_df.iloc[:valid_n].reset_index()
    grouped_kernel_df = test_df.groupby(test_df.index // kernels_num, sort=False).agg(
        {"self_device_time_total": "sum", "index": list}
    )

    # rm warm iters
    sum_df = grouped_kernel_df.iloc[warm_iter:].reset_index(drop=True)
    out_range_idx = []
    if num_iters > 30:
        # IQR to remove abnormal data
        k = 1.5
        Q1 = sum_df["self_device_time_total"].quantile(0.25)
        Q3 = sum_df["self_device_time_total"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        out_range_idx = sum_df.index[
            (sum_df["self_device_time_total"] < lower)
            | (sum_df["self_device_time_total"] > upper)
        ].tolist()
    out_range_num = len(out_range_idx)

    indices = {idx for i in out_range_idx for idx in sum_df.iloc[i]["index"]}

    index_sublists = grouped_kernel_df["index"].head(warm_iter).tolist()
    indices_to_add = [idx for sublist in index_sublists for idx in sublist]
    indices.update(indices_to_add)
    indices.update(dropped_indexs)
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
    df = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    ###remove abnormal data
    dropped_num = warm_iter
    dropped_indexs, dropped_num = post_process_data(
        df, num_iters + warm_iter, warm_iter
    )
    df = df.drop(dropped_indexs)
    iter_init = 0  # warm_iter dropped
    df["cnt"] = 1
    rets = []

    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][
            ["cnt", "self_cpu_time_total", "self_device_time_total"]
        ].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0
        rets.append(r)
    df = pd.DataFrame(rets)
    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]

    timerList = [
        "host_time_sum",
        "device_time_sum",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    actual_iters = num_iters + warm_iter - dropped_num
    if df.empty:
        logger.info("no valida data after post process!")

    avg_name = "[avg us/iter]"
    for el in timerList:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    return df.at[avg_name, "device_time_sum"]
