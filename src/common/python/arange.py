"""
multi_arange：将多段 [start, end) 区间合并为一个连续 tensor。

来源统一：MTGraph/src/framework/Operators/operators.py 的实现采用
torch.arange + repeat_interleave，比朴素循环更快。
"""
import torch


def multi_arange(starts, ends):
    """
    合并多段 [starts[i], ends[i]) 区间为单个 tensor。

    Args:
        starts: 1D tensor，每段起点
        ends:   1D tensor，每段终点（同 shape、同 device）

    Returns:
        1D tensor，长度 sum(ends - starts)，按段顺序拼接
    """
    device = starts.device
    sizes = ends - starts
    begin_idx = sizes.cumsum(0)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), begin_idx])
    begin_idx = ptr[:-1]
    result = torch.arange(ptr[-1], device=device) - (begin_idx - starts).repeat_interleave(sizes)
    return result
