"""
图分区工具。

包含两个 API：
- EBP(row_ptr, columns, partition_num)
    返回 (parts_row_ptr, parts_columns, vertex_begin_idx)
    Edge-Balanced Partition：按度数累加做边均衡划分（与 CUDA 端 ebp() 一致）。
- get_partition(row_ptr, columns, partition_num)
    同 EBP（兼容旧脚本调用名）。

来源：MTGraph/src/quicktest/help.py 的 Python 版 EBP；
逻辑与 MTGraph/src/bfs/bfs_multigpu.cu 的 ebp() 等价。
"""
import torch


def EBP(row_ptr, columns, partition_num):
    """
    Edge-Balanced Partition（边均衡划分）。

    Args:
        row_ptr: CSR 行指针（1D tensor）
        columns: CSR 列索引（1D tensor）
        partition_num: 分区数

    Returns:
        parts_row_ptr:   list of tensor，每个分区的本地 row_ptr（int64）
        parts_columns:   list of tensor，每个分区的本地 columns（int32）
        vertex_begin_idx: tensor，每个分区的起始顶点索引（长度 partition_num + 1）
    """
    row_ptr = row_ptr.to(torch.long)
    degrees = torch.diff(row_ptr)
    degrees = degrees.to(torch.long)

    degrees_sum = torch.cumsum(degrees, dim=0, dtype=torch.int64)
    all_sum = degrees_sum[-1]

    partition_size = all_sum // partition_num
    target_sizes = [i * partition_size for i in range(1, partition_num)]
    target_sizes = torch.tensor(target_sizes, dtype=torch.int64)
    indexs = torch.searchsorted(degrees_sum, target_sizes)
    vertex_begin_idx = [0]

    for i in range(partition_num - 1):
        vertex_begin_idx.append(indexs[i].item() + 1)

    vertex_begin_idx.append(len(degrees))
    vertex_begin_idx = torch.tensor(vertex_begin_idx, dtype=torch.int64)

    parts_row_ptr = []
    parts_columns = []

    for i in range(partition_num):
        part_degrees = degrees[vertex_begin_idx[i]:vertex_begin_idx[i + 1]]
        part_row_ptr = torch.cumsum(part_degrees, dim=0)
        part_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int64), part_row_ptr])

        col_begin = row_ptr[vertex_begin_idx[i]]
        col_end = row_ptr[vertex_begin_idx[i + 1]]
        part_columns = columns[col_begin:col_end]

        # part_row_ptr 保持 int64：避免单分区边数 > 2^31 (21亿) 时溢出
        # 下游 segment_csr 本就要求 ptr 为 int64，bfs.py/pr.py 的调用无需再转换
        part_row_ptr = part_row_ptr.to(torch.int64)
        # columns 存顶点 ID（< 21 亿），int32 足够
        part_columns = part_columns.to(torch.int32)
        parts_row_ptr.append(part_row_ptr)
        parts_columns.append(part_columns)

    return parts_row_ptr, parts_columns, vertex_begin_idx


# 别名（兼容 TGraph 脚本里的 get_partition 调用名）
get_partition = EBP
