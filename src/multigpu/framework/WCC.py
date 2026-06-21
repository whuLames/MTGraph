"""多 GPU 框架版 WCC（GASProgram + DistributedStrategy）。"""
import os; import torch.distributed as dist; import argparse
from src.framework.strategy.DistributedStrategy import DistributedStrategy

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', '-p', type=str, required=True)
    p.add_argument('-n', '--partition_num', type=int, default=2)
    p.add_argument('-l', '--long', type=int, default=0)
    p.add_argument('-i', '--max_iters', type=int, default=50)
    args = p.parse_args()
    rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    s = DistributedStrategy.from_program('ConnectedComponents', data_path=args.data_path,
                                          partition_num=args.partition_num, is_long=args.long,
                                          max_iters=args.max_iters)
    vData, stats = s.compute()
    if rank == 0:
        print(f'Rank 0 components {vData.unique().numel()} iters {stats["iters"]} '
              f'one-iter time {stats["per_iter"]}')

if __name__ == '__main__': main()
