"""
纯 torch 多 GPU SSSP（单源最短路径，边权重=1）。

边权重固定为 1，SSSP 退化为带 float 距离的 BFS（pull 模式）。
通信：torch.distributed NCCL。
读图 / 分区：复用 src.common.python（分布式读图 + EBP）。
"""
import os as _os; _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import os
import time
from torch_scatter import segment_csr
import argparse

from src.common.python.distributed import distributed_read_and_partition


def sssp(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--iterations', '-i', type=int, default=50)
    parser.add_argument('--source', '-s', type=int, default=0)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()
    data_path = args.data_path
    partition_num = args.partition_num
    max_iters = args.iterations
    source = args.source
    is_long = args.long

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    local_rank = rank % torch.cuda.device_count()
    device = f'cuda:{local_rank}'

    row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts = \
        distributed_read_and_partition(data_path, partition_num, is_long=is_long)
    if rank == 0:
        print(f'v {n_verts} (distributed read done)')

    part_row_ptr = part_row_ptr.to(device).to(torch.int64)
    part_columns = part_columns.to(device).to(torch.int64)

    # init: dist[source] = 0, 其余 inf
    vData = torch.full((n_verts,), float('inf'), dtype=torch.float32, device=device)
    vData[source] = 0.0

    t1 = time.time()
    for i in range(max_iters):
        aggData = vData[part_columns] + 1.0
        aggRes = segment_csr(aggData, part_row_ptr, reduce='min')
        del aggData
        local_old = vData[vertex_begin:vertex_end]
        local_new = torch.min(aggRes, local_old)
        changed = (local_new < local_old).sum()
        vData[vertex_begin:vertex_end] = local_new
        del aggRes, local_old, local_new
        dist.all_reduce(vData, op=dist.ReduceOp.MIN)
        changed_tensor = torch.tensor([changed.item()], device=device, dtype=torch.int64)
        dist.all_reduce(changed_tensor, op=dist.ReduceOp.SUM)
        if changed_tensor.item() == 0:
            if rank == 0:
                print(f'converged at iter {i + 1}')
            break

    t2 = time.time()
    if rank == 0:
        per_iter = (t2 - t1) / (i + 1)
        max_dist = vData[vData < float('inf')].max().item()
        reachable = (vData < float('inf')).sum().item()
        print(f'Rank {rank} maxdist {max_dist} reachable {reachable} '
              f'iters {i + 1} one-iter time {per_iter}')


if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    sssp(rank, world_size)
