"""
纯 torch 多 GPU WCC（弱连通分量，Label Propagation）。

通信：torch.distributed NCCL。
读图 / 分区：复用 src.common.python（分布式读图 + EBP）。
假设图为无向（CSR 已包含双向边）。
"""
import os as _os; _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import os
import time
from torch_scatter import segment_csr
import argparse

from src.common.python.distributed import distributed_read_and_partition


def wcc(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--iterations', '-i', type=int, default=50)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()
    data_path = args.data_path
    partition_num = args.partition_num
    max_iters = args.iterations
    is_long = args.long

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    local_rank = rank % torch.cuda.device_count()
    device = f'cuda:{local_rank}'

    row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts = \
        distributed_read_and_partition(data_path, partition_num, is_long=is_long)
    if rank == 0:
        print(f'v {n_verts} (distributed read done)')

    part_row_ptr = part_row_ptr.to(device).to(torch.int64)  # 预转 int64，避免每轮 cast
    part_columns = part_columns.to(device).to(torch.int64)  # 预转 int64，避免 fancy indexing 每轮临时 cast

    # init: label[i] = i
    vData = torch.arange(n_verts, dtype=torch.int32, device=device)

    t1 = time.time()
    for i in range(max_iters):
        aggData = vData[part_columns]
        aggRes = segment_csr(aggData, part_row_ptr, reduce='min')
        del aggData  # 立即释放临时 tensor
        # 只更新本地顶点
        local_old = vData[vertex_begin:vertex_end]
        local_new = torch.min(aggRes, local_old)
        changed = (local_new < local_old).sum()
        vData[vertex_begin:vertex_end] = local_new
        del aggRes, local_old, local_new
        # 全局同步
        dist.all_reduce(vData, op=dist.ReduceOp.MIN)
        # 收敛检查（所有 rank 的 changed 之和为 0）
        changed_tensor = torch.tensor([changed.item()], device=device, dtype=torch.int64)
        dist.all_reduce(changed_tensor, op=dist.ReduceOp.SUM)
        if changed_tensor.item() == 0:
            if rank == 0:
                print(f'converged at iter {i + 1}')
            break

    t2 = time.time()
    if rank == 0:
        per_iter = (t2 - t1) / (i + 1)
        # 统计连通分量数
        num_components = torch.unique(vData).numel()
        print(f'Rank {rank} components {num_components} iters {i + 1} '
              f'one-iter time {per_iter}')


if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    wcc(rank, world_size)
