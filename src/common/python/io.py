"""
图数据 I/O 工具。
统一两项目的 read_data / readGraphBinary 实现。

数据格式约定：
- <path>/csr_vlist.bin  : CSR 行指针（int32，或 is_long=True 时 int64）
- <path>/csr_elist.bin  : CSR 列索引（int32）
- <path>/csr_wlist.bin  : 边权重（float32，仅 is_weighted=True）
"""
import os
import numpy as np
import torch


def read_data(path, is_weighted=False, is_long=False):
    """
    从目录读取 CSR 格式图（两份文件 csr_vlist.bin + csr_elist.bin）。

    Args:
        path: 图数据目录
        is_weighted: 是否读取权重文件 csr_wlist.bin
        is_long: True 时 row_ptr 按 int64 读取（大图用）

    Returns:
        row_ptr, columns[, weights]  —— torch.Tensor（CPU）
    """
    path_v = os.path.join(path, "csr_vlist.bin")
    path_e = os.path.join(path, "csr_elist.bin")

    row_ptr = np.fromfile(path_v, dtype=np.int64 if is_long else np.int32)
    columns = np.fromfile(path_e, dtype=np.int32)

    row_ptr = torch.from_numpy(row_ptr)
    columns = torch.from_numpy(columns)

    if is_weighted:
        path_w = os.path.join(path, "csr_wlist.bin")
        weights = torch.from_numpy(np.fromfile(path_w, dtype=np.float32))
        return row_ptr, columns, weights

    return row_ptr, columns


def readGraphBinary(path):
    """
    兼容旧 API：默认 int32，返回 (row_ptr, columns)。
    等价于 read_data(path, is_weighted=False, is_long=False)。
    """
    return read_data(path, is_weighted=False, is_long=False)
