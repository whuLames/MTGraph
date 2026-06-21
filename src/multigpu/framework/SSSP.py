"""多 GPU 框架版 SSSP（GASProgram + DistributedStrategy, 边权重=1）。"""
import os; import torch.distributed as dist; import argparse
from src.framework.strategy.DistributedStrategy import DistributedStrategy

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', '-p', type=str, required=True)
    p.add_argument('-n', '--partition_num', type=int, default=2)
    p.add_argument('-s', '--source', type=int, default=0)
    p.add_argument('-l', '--long', type=int, default=0)
    p.add_argument('-i', '--max_iters', type=int, default=50)
    args = p.parse_args()
    rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    s = DistributedStrategy.from_program('SSSP', data_path=args.data_path,
                                          partition_num=args.partition_num, is_long=args.long,
                                          max_iters=args.max_iters, source=args.source)
    vData, stats = s.compute()
    if rank == 0:
        import math as m
        visited = (vData < m.inf).sum().item()
        print(f'Rank 0 reachable {visited} iters {stats["iters"]} '
              f'one-iter time {stats["per_iter"]}')

if __name__ == '__main__': main()
