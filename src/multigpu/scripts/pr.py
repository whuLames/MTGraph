"""
纯 torch 多 GPU PageRank（来自 TGraph/nonsingle/distributed/torch/pr.py）。

通信：torch.distributed NCCL。
读图 / 分区 / 区间合并：复用 src.common.python。
"""
import os as _os; _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import os
import time
from torch_scatter import segment_csr
import argparse

# 公共模块（pip install -e . 后可用）
from src.common.python.io import read_data
from src.common.python.partition import get_partition
from src.common.python.distributed import distributed_read_and_partition


def pr(rank, world_size):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # get paras
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--iterations', '-i', type=int, default=10)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()
    data_path = args.data_path
    partition_num = args.partition_num
    iters = args.iterations
    is_long = args.long

    print('data_path: ', data_path, 'partition_num: ', partition_num, 'iterations: ', iters)
    # init process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    

    local_rank = rank % torch.cuda.device_count()
    print(f'Rank {rank} has device cuda:{local_rank}')
    device = f'cuda:{local_rank}'

    # 分布式读图 + EBP 分区（rank 0 读图，broadcast row_ptr + scatter 分区）
    row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts = \
        distributed_read_and_partition(data_path, partition_num, is_long=is_long)
    if rank == 0:
        print(f'v {n_verts} (distributed read done)')

    part_row_ptr = part_row_ptr.to(device)
    part_columns = part_columns.to(device)
    vData = torch.ones(n_verts, dtype=torch.float32, device=device)  # 全局节点数据

    frac = 1 / torch.diff(row_ptr).to(torch.float32)
    frac = frac.to(device)

    t1 = time.time()
    # do computation
    for i in range(iters):
        aggData = vData * frac
        aggData = aggData[part_columns]
        # segment_csr 要求 ptr 为 int64（PyTorch 2.x + torch_scatter 兼容）
        aggRes = segment_csr(aggData, part_row_ptr.to(torch.int64), reduce='sum')
        vData.zero_()
        vData[vertex_begin:vertex_end] = aggRes # 除了本地节点，其他节点值均为zero
        
        ta = time.time()
        dist.all_reduce(vData, op=dist.ReduceOp.SUM)
        tb = time.time()
        if i == iters - 2:
            print(f'Rank {rank} all reduce time {tb - ta}')
        del aggData, aggRes
    t2 = time.time()

    sum = torch.sum(vData)
    print(f'Rank {rank} has data {sum} after {iters} all reduce one-iter time {(t2 - t1) / iters}')

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # print('The rank in env is: ', rank)
    # allreduce(rank, world_size)
    pr(rank, world_size)