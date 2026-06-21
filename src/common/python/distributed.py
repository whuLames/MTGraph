"""
torch.distributed NCCL 辅助工具。

来源统一：TGraph/nonsingle/distributed/torch/{bfs,pr}.py 的 allreduce +
init_process_group 模式。
"""
import os
import torch
import torch.distributed as dist


def init_nccl(rank=None, world_size=None, init_method="env://"):
    """
    初始化 NCCL 进程组。

    优先用环境变量 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT；
    若未设置，则用显式传入的 rank / world_size。
    """
    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    return local_rank


def allreduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """
    对 tensor 做 all-reduce。op 默认 SUM，可传 MIN/MAX/PRODUCT。
    """
    dist.all_reduce(tensor, op=op)
    return tensor


def get_local_rank():
    """从环境变量推断 local rank 并 set_device。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return local_rank


def distributed_read_and_partition(data_path, partition_num, is_long=False):
    """
    Rank 0 读完整图 + EBP 分区，broadcast 全局 row_ptr + scatter 各分区。

    与 "每 rank 各读全量图" 相比：
      - 磁盘 I/O 从 N 份降为 1 份
      - 内存从 N × full 降为 1 × full + N × (1/N)
      - 对迭代性能零影响（预处理只做一次）

    在 ``dist.init_process_group`` 之后调用。所有 rank 同时调用。

    Args:
        data_path:    图数据目录（csr_vlist.bin + csr_elist.bin）
        partition_num: 分区数（必须 == world_size）
        is_long:      True 时 row_ptr 按 int64 读取（边数 > 2^31 用）

    Returns:
        row_ptr:        全局 CSR row_ptr（int64, CPU），所有 rank 相同
        part_row_ptr:   本地分区 row_ptr（int64, CPU）
        part_columns:   本地分区 columns（int32, CPU）
        vertex_begin:   本地分区起始全局顶点 ID
        vertex_end:     本地分区结束全局顶点 ID
        n_verts:        全局顶点数
    """
    from src.common.python.io import read_data
    from src.common.python.partition import EBP

    assert dist.is_initialized(), "distributed_read_and_partition 需在 init_process_group 之后调用"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert partition_num == world_size, f"partition_num({partition_num}) != world_size({world_size})"

    # NCCL backend 仅支持 GPU tensor；读图/分区在 CPU 上做，需 gloo backend 的 group
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend='gloo')

    # ---- Step 1: rank 0 读图 + EBP 分区 ----
    if rank == 0:
        row_ptr_raw, columns_raw = read_data(data_path, is_long=is_long)
        row_ptr_raw = row_ptr_raw.to(torch.int64)
        parts_rp, parts_col, vbidx = EBP(row_ptr_raw, columns_raw, partition_num)
        row_ptr = row_ptr_raw
        parts_row_ptr = parts_rp
        parts_columns = parts_col
        vb = vbidx
    else:
        row_ptr = None
        parts_row_ptr = None
        parts_columns = None
        vb = None

    # ---- Step 2: broadcast 全局 row_ptr ----
    if rank == 0:
        n_meta = torch.tensor([len(row_ptr)], dtype=torch.int64)
    else:
        n_meta = torch.zeros(1, dtype=torch.int64)
    dist.broadcast(n_meta, src=0, group=cpu_group)
    rp_len = n_meta.item()

    if rank != 0:
        row_ptr = torch.empty(rp_len, dtype=torch.int64)
    dist.broadcast(row_ptr, src=0, group=cpu_group)
    n_verts = rp_len - 1

    # ---- Step 3: broadcast vertex_begin_idx ----
    if rank != 0:
        vb = torch.empty(partition_num + 1, dtype=torch.int64)
    dist.broadcast(vb, src=0, group=cpu_group)
    vertex_begin = vb[rank].item()
    vertex_end = vb[rank + 1].item()

    # ---- Step 4: broadcast 各分区 sizes ----
    if rank == 0:
        sizes = torch.tensor([[len(rp), len(c)] for rp, c in zip(parts_row_ptr, parts_columns)],
                             dtype=torch.int64)
    else:
        sizes = torch.empty(partition_num, 2, dtype=torch.int64)
    dist.broadcast(sizes, src=0, group=cpu_group)

    my_rp_len = sizes[rank][0].item()
    my_col_len = sizes[rank][1].item()

    # ---- Step 5: point-to-point send/recv 本地分区 ----
    if rank == 0:
        part_row_ptr = parts_row_ptr[0]
        part_columns = parts_columns[0]
        for dst in range(1, world_size):
            dist.send(parts_row_ptr[dst].contiguous(), dst=dst, group=cpu_group)
            dist.send(parts_columns[dst].contiguous(), dst=dst, group=cpu_group)
    else:
        part_row_ptr = torch.empty(my_rp_len, dtype=torch.int64)
        part_columns = torch.empty(my_col_len, dtype=torch.int32)
        dist.recv(part_row_ptr, src=0, group=cpu_group)
        dist.recv(part_columns, src=0, group=cpu_group)

    return row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts
