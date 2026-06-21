"""
基于 DistributedStrategy 的统一多 GPU 入口。

用 DistributedStrategy 跑 PR / BFS / WCC / SSSP，验证：
  1. 功能正确性（与脚本版结果一致）
  2. 性能（与脚本版对比，不应下降）

用法：
  torchrun --nproc_per_node=4 src/multigpu/framework/distributed_entry.py \
      --data_path /path/to/graph --algorithm pr -n 4 -i 10
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import argparse
import time

from src.framework.strategy.DistributedStrategy import DistributedStrategy


def run_pr(args):
    """PageRank via DistributedStrategy."""
    frac_cache = {}
    def prepare_fn(row_ptr, device):
        frac = 1.0 / torch.diff(row_ptr).to(torch.float32).to(device)
        return {'frac': frac, 'n_verts': len(row_ptr) - 1}

    def init_fn(n, d):
        return torch.ones(n, dtype=torch.float32, device=d)

    def gather_fn(vd, col, ctx):
        # 先全局 vData * frac，再按边索引（与脚本版 pr.py 一致）
        return (vd * ctx['frac'])[col]

    def apply_fn(old, agg, ctx):
        return 0.15 / ctx['n_verts'] + 0.85 * agg

    strategy = DistributedStrategy(
        data_path=args.data_path, partition_num=args.partition_num,
        reduce='sum', allreduce_op=dist.ReduceOp.SUM,
        init_fn=init_fn, gather_fn=gather_fn, apply_fn=apply_fn,
        prepare_fn=prepare_fn,
        max_iters=args.iterations, fixed_iters=True,
        is_long=args.long,
    )
    return strategy.compute()


def run_bfs(args):
    """BFS via DistributedStrategy (supports push/pull when enabled)."""
    INF = 10000.0

    def init_fn(n, d):
        vd = torch.full((n,), INF, dtype=torch.float32, device=d)
        vd[args.source] = 0.0
        return vd

    def gather_fn(vd, col, ctx):
        return vd[col] + 1.0

    def apply_fn(old, agg, ctx):
        return torch.min(old, agg)

    # push/pull 回调（仅在 enable_push_pull 时使用）
    def frontier_fn(vd, vb, ve, ctx):
        """返回本地活跃顶点（本轮有更新的顶点）。"""
        old = vd[vb:ve]
        return torch.nonzero(old < ctx['INF']).view(-1)

    def push_value_fn(vd, frontier_global, ctx):
        """每个 frontier 顶点要推送的值 = dist + 1。"""
        return vd[frontier_global] + 1.0

    strategy = DistributedStrategy(
        data_path=args.data_path, partition_num=args.partition_num,
        reduce='min', allreduce_op=dist.ReduceOp.MIN,
        init_fn=init_fn, gather_fn=gather_fn, apply_fn=apply_fn,
        max_iters=args.iterations, fixed_iters=False,
        is_long=args.long, source=args.source,
        # push/pull 配置
        enable_push_pull=getattr(args, 'push_pull', False),
        push_threshold_ratio=1.0/15,
        frontier_fn=frontier_fn,
        push_value_fn=push_value_fn,
        prepare_fn=lambda row_ptr, device: {'INF': INF},
    )
    v_data, stats = strategy.compute()

    rank = dist.get_rank()
    if rank == 0:
        mask = v_data < INF
        max_dist = v_data[mask].max().item() if mask.any() else 0
        reachable = mask.sum().item()
        print(f'Rank {rank} maxdist {max_dist} reachable {reachable} '
              f'iters {stats["iters"]} one-iter time {stats["per_iter"]}')
    return v_data, stats


def run_wcc(args):
    """WCC (Label Propagation) via DistributedStrategy."""
    def init_fn(n, d):
        return torch.arange(n, dtype=torch.int32, device=d)

    def gather_fn(vd, col, ctx):
        return vd[col]

    def apply_fn(old, agg, ctx):
        return torch.min(old, agg)

    strategy = DistributedStrategy(
        data_path=args.data_path, partition_num=args.partition_num,
        reduce='min', allreduce_op=dist.ReduceOp.MIN,
        init_fn=init_fn, gather_fn=gather_fn, apply_fn=apply_fn,
        max_iters=args.iterations, fixed_iters=False,
        is_long=args.long,
    )
    v_data, stats = strategy.compute()

    rank = dist.get_rank()
    if rank == 0:
        num_components = torch.unique(v_data).numel()
        print(f'Rank {rank} components {num_components} '
              f'iters {stats["iters"]} one-iter time {stats["per_iter"]}')
    return v_data, stats


def run_sssp(args):
    """SSSP (edge weight=1) via DistributedStrategy."""
    def init_fn(n, d):
        vd = torch.full((n,), float('inf'), dtype=torch.float32, device=d)
        vd[args.source] = 0.0
        return vd

    def gather_fn(vd, col, ctx):
        return vd[col] + 1.0

    def apply_fn(old, agg, ctx):
        return torch.min(old, agg)

    strategy = DistributedStrategy(
        data_path=args.data_path, partition_num=args.partition_num,
        reduce='min', allreduce_op=dist.ReduceOp.MIN,
        init_fn=init_fn, gather_fn=gather_fn, apply_fn=apply_fn,
        max_iters=args.iterations, fixed_iters=False,
        is_long=args.long, source=args.source,
    )
    v_data, stats = strategy.compute()

    rank = dist.get_rank()
    if rank == 0:
        mask = v_data < float('inf')
        reachable = mask.sum().item()
        print(f'Rank {rank} reachable {reachable} '
              f'iters {stats["iters"]} one-iter time {stats["per_iter"]}')
    return v_data, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str, required=True)
    parser.add_argument('--algorithm', '-a', type=str, required=True,
                        choices=['pr', 'bfs', 'wcc', 'sssp'])
    parser.add_argument('--partition_num', '-n', type=int, required=True)
    parser.add_argument('--iterations', '-i', type=int, default=50)
    parser.add_argument('--source', '-s', type=int, default=0)
    parser.add_argument('--long', '-l', type=int, default=0)
    parser.add_argument('--push_pull', action='store_true',
                        help='Enable push/pull adaptive switching (BFS only)')
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    dispatch = {'pr': run_pr, 'bfs': run_bfs, 'wcc': run_wcc, 'sssp': run_sssp}
    dispatch[args.algorithm](args)

    dist.destroy_process_group()
