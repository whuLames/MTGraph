"""
多 GPU 框架版 BFS（基于 GASProgram + DistributedStrategy）。

算法语义由 src/algorithms/BFS.py 的 distributed_config() 提供，
DistributedStrategy 自动适配为分布式调度。
启动方式：BFS_MODE=single torchrun --nproc_per_node=N ... --data_path <path> -n N -s 0
"""
import os
import torch
import torch.distributed as dist
import argparse
from src.framework.strategy.DistributedStrategy import DistributedStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str, required=True,
                        help='path to graph dir (csr_vlist[.bin] / csr_elist[.bin])')
    parser.add_argument('-n', '--partition_num', type=int, default=2)
    parser.add_argument('-s', '--source', type=int, default=0)
    parser.add_argument('-l', '--long', type=int, default=0)
    parser.add_argument('-i', '--max_iters', type=int, default=50)
    parser.add_argument('--push_pull', action='store_true',
                        help='enable push/pull adaptive switching')
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    strategy = DistributedStrategy.from_program(
        'BFS', data_path=args.data_path, partition_num=args.partition_num,
        is_long=args.long, max_iters=args.max_iters,
        enable_push_pull=args.push_pull, source=args.source)

    vData, stats = strategy.compute()
    if rank == 0:
        import math
        visited = vData[vData < 1e4].numel()
        maxv = vData[vData < 1e4].max().item()
        print(f'Rank {rank} maxdist {maxv:.0f} reachable {visited} '
              f'iters {stats["iters"]} one-iter time {stats["per_iter"]}')


if __name__ == '__main__':
    main()
