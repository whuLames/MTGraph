#!/usr/bin/env python
"""
多节点 all-reduce 基准测试。

用法（双节点 × 3 卡 = 6 rank）：

节点 0（master）：
  torchrun --nnodes=2 --nproc_per_node=3 \
      --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:29500 \
      allreduce_benchmark.py

节点 1（worker）：
  torchrun --nnodes=2 --nproc_per_node=3 \
      --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:29500 \
      allreduce_benchmark.py

单节点测试（6 卡）：
  torchrun --nproc_per_node=6 allreduce_benchmark.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import time
import argparse


def benchmark_allreduce(tensor_len, warmup=5, iters=20):
    """
    对指定大小的 float32 tensor 做 all-reduce 基准测试。

    Args:
        tensor_len: tensor 元素个数
        warmup:     预热轮次
        iters:      计时轮次

    Returns:
        avg_ms:  平均每轮时间（毫秒）
        bandwidth_gbps: 有效带宽（GB/s），= data_size / avg_time
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank % torch.cuda.device_count()}'

    # 创建 tensor
    tensor = torch.randn(tensor_len, dtype=torch.float32, device=device)

    # warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # 计时
    t1 = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    t2 = time.time()

    avg_s = (t2 - t1) / iters
    avg_ms = avg_s * 1000

    # 带宽：ring all-reduce 通信量 ≈ 2*(N-1)/N * data_size
    data_size_bytes = tensor_len * 4  # float32
    comm_bytes = 2 * (world_size - 1) / world_size * data_size_bytes
    bandwidth_gbps = comm_bytes / avg_s / 1e9

    return avg_ms, bandwidth_gbps


def main():
    parser = argparse.ArgumentParser(description='All-Reduce Benchmark')
    parser.add_argument('--warmup', type=int, default=5, help='warmup iterations')
    parser.add_argument('--iters', type=int, default=20, help='timed iterations')
    parser.add_argument('--sizes', type=str, default='20M,50M,100M',
                        help='tensor sizes, comma-separated (e.g. 20M,50M,100M)')
    args = parser.parse_args()

    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank % torch.cuda.device_count()}'
    torch.cuda.set_device(rank % torch.cuda.device_count())

    if rank == 0:
        print(f'')
        print(f'===== All-Reduce Benchmark =====')
        print(f'World size:  {world_size}')
        print(f'Nodes:       {world_size // 3} (assuming 3 GPUs/node)')
        print(f'Warmup:      {args.warmup} iters')
        print(f'Timed:       {args.iters} iters')
        print(f'Backend:     NCCL')
        print(f'')

    dist.barrier()

    # 解析 tensor 大小
    size_map = {'K': 1e3, 'M': 1e6, 'B': 1e9}
    sizes = []
    for s in args.sizes.split(','):
        s = s.strip().upper()
        for suffix in ['M', 'K', 'B']:
            if s.endswith(suffix):
                sizes.append(int(float(s[:-1]) * size_map[suffix]))
                break
        else:
            sizes.append(int(s))

    # 跑基准
    results = []
    for tensor_len in sizes:
        data_mb = tensor_len * 4 / 1e6
        avg_ms, bw = benchmark_allreduce(tensor_len, args.warmup, args.iters)

        results.append((tensor_len, data_mb, avg_ms, bw))

        if rank == 0:
            print(f'  Size: {tensor_len:>12,} elements ({data_mb:>8.1f} MB)  |  '
                  f'Avg time: {avg_ms:>8.2f} ms  |  '
                  f'Bandwidth: {bw:>6.1f} GB/s')

    # 汇总表格
    if rank == 0:
        print(f'')
        print(f'{"─"*75}')
        print(f'{"Size (elements)":>16}  {"Data (MB)":>10}  {"Avg (ms)":>10}  {"BW (GB/s)":>10}')
        print(f'{"─"*75}')
        for t_len, data_mb, avg_ms, bw in results:
            print(f'{t_len:>16,}  {data_mb:>10.1f}  {avg_ms:>10.2f}  {bw:>10.1f}')
        print(f'{"─"*75}')
        print(f'')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
