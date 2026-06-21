"""
纯 torch 多 GPU BFS（来自 TGraph/nonsingle/distributed/torch/bfs.py）。

含两个变体：
- bfs()     : 单层 push/pull 自适应 + 每轮 all_reduce(MIN)
- bfs_bc()  : 内层 while True 本地多轮收敛 + 一次 all_reduce(MIN)（direction-optimizing）

通信：torch.distributed NCCL。
读图 / 分区 / 区间合并：复用 src.common.python。
"""
import os as _os; _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import os
import time
from torch_scatter import segment_csr, scatter
import argparse

# 公共模块（pip install -e . 后可用）
from src.common.python.io import read_data
from src.common.python.partition import get_partition
from src.common.python.arange import multi_arange
from src.common.python.distributed import distributed_read_and_partition


def bfs(rank, world_size):
    """
    BFS vc
    """
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # get paras
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--source', '-s', type=int, default=0)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()

    data_path = args.data_path
    partition_num = args.partition_num
    source = args.source
    is_long = args.long

    print('data_path: ', data_path, 'partition_num: ', partition_num, 'source: ', source)
    # init process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    

    local_rank = rank % torch.cuda.device_count()
    
    # do graph partition
    row_ptr, columns = read_data(data_path, is_long=is_long)
    print(f'v {len(row_ptr) - 1} e {len(columns)}')
    parts_row_ptr, parts_columns, vertex_begin_idx = get_partition(row_ptr, columns, partition_num)
    device = f'cuda:{local_rank}'
    part_row_ptr = parts_row_ptr[rank]
    part_columns = parts_columns[rank]
    INF = 10000

    # init data
    vData = torch.zeros(len(row_ptr) - 1, dtype=torch.float32, device=device)  + INF # 全局节点数据 
    vData[source] = 0
    actMask = torch.zeros(len(row_ptr) - 1, dtype=torch.int32, device=device)  # 活跃节点标记
    actMask[source] = 1
    modeSwitchThreshold = len(part_columns) // 15
    part_row_ptr = part_row_ptr.to(device)
    part_columns = part_columns.to(device)
    vertex_begin = vertex_begin_idx[rank]
    vertex_end = vertex_begin_idx[rank + 1]
    part_degrees = torch.diff(part_row_ptr)
    iter = 0
    aggData = torch.zeros_like(vData) # 用来保存每个subgraph的本地计算结果

    t1 = time.time()
    while torch.any(actMask):
        compute_time = 0
        allreduce_time = 0

        ta = time.time()
        aggData[:] = vData[:] # 保存上一轮的结果
        actLocalMask = actMask[vertex_begin : vertex_end]
        frontierLocal = torch.nonzero(actLocalMask).view(-1)
        numEdgesToProcess = torch.sum(part_degrees[frontierLocal])
        tb = time.time()
        compute_time += tb - ta

        ta = time.time()
        if numEdgesToProcess == 0:
            print('rank {} iter {} numEdgesToProcess is 0'.format(rank, iter))
        
        elif numEdgesToProcess < modeSwitchThreshold:
            print('rank {} iter {} do Push computation'.format(rank, iter, numEdgesToProcess))
            starts = part_row_ptr[frontierLocal]
            ends = part_row_ptr[frontierLocal + 1]
            neighborIndices = multi_arange(starts, ends)
            neighbors = torch.index_select(part_columns, 0, neighborIndices)
            neighbors = neighbors.to(torch.int64)
            aggData[neighbors] = iter + 1
            del neighborIndices, neighbors
        else:
            print('rank {} iter {} do Pull computation'.format(rank, iter, numEdgesToProcess))
            groupData =  torch.index_select(vData, 0, part_columns)
            groupData.add_(1)
            groupID = part_row_ptr
            # segment_csr 要求 ptr 为 int64（PyTorch 2.x + torch_scatter 兼容）
            aggRes = segment_csr(groupData, groupID.to(torch.int64), reduce='min')
            aggData[vertex_begin : vertex_end] = aggRes
            del groupData, groupID, aggRes
        tb = time.time()
        compute_time += tb - ta
        # min all reduce to get the global result
        ta = time.time()
        dist.all_reduce(aggData, op=dist.ReduceOp.MIN)
        tb = time.time()
        print(f'rank{rank} compute time {compute_time}  all reduce time {tb-ta} in iter {iter} ')
        actMask = aggData < vData
        vData = torch.min(aggData, vData)
        iter += 1
    t2 = time.time()
    maxVal = torch.max(vData)
    mask = vData == INF
    nonVisitedNum = torch.sum(mask) 
    print('rank {} maxVal {} nonVisitedNum {}'.format(rank, iter, nonVisitedNum))
    print('vData: ', vData[0:40], 'elapsed time: ', t2 - t1)


def bfs_single(rank, world_size):
    """
    BFS 单层 push/pull 自适应（无内层多轮本地松弛）。
    每轮做一次 push 或 pull + all_reduce(MIN)。
    通过环境变量 BFS_MODE=single 选择此模式。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--source', '-s', type=int, default=0)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()
    data_path = args.data_path
    partition_num = args.partition_num
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

    INF = 10000
    part_row_ptr = part_row_ptr.to(device).to(torch.int64)
    part_columns = part_columns.to(device).to(torch.int64)

    vData = torch.full((n_verts,), float(INF), dtype=torch.float32, device=device)
    vData[source] = 0.0
    actMask = torch.zeros(n_verts, dtype=torch.bool, device=device)
    actMask[source] = True
    modeSwitchThreshold = len(part_columns) // 15
    part_degrees = torch.diff(part_row_ptr)

    t1 = time.time()
    iter_num = 0
    while torch.any(actMask):
        aggData = vData.clone()
        # 本地 frontier
        frontierLocal = torch.nonzero(actMask[vertex_begin:vertex_end]).view(-1)
        if len(frontierLocal) == 0:
            dist.all_reduce(aggData, op=dist.ReduceOp.MIN)
            actMask = aggData < vData
            vData = torch.min(aggData, vData)
            iter_num += 1
            continue

        numEdgesToProcess = torch.sum(part_degrees[frontierLocal]).item()

        if numEdgesToProcess < modeSwitchThreshold:
            # ---- Push 模式 ----
            frontierGlobal = frontierLocal + vertex_begin
            starts = part_row_ptr[frontierLocal]
            ends = part_row_ptr[frontierLocal + 1]
            neighborIndices = multi_arange(starts, ends)
            neighbors = part_columns[neighborIndices]
            groupData = (vData[frontierGlobal] + 1).repeat_interleave(part_degrees[frontierLocal])
            aggRes = scatter(groupData, neighbors, dim_size=n_verts, reduce='min')
            del groupData, neighbors, neighborIndices, frontierGlobal
            mask = aggRes > 0
            aggData[mask] = aggRes[mask]
            del aggRes, mask
        else:
            # ---- Pull 模式 ----
            groupData = vData[part_columns] + 1.0
            aggRes = segment_csr(groupData, part_row_ptr, reduce='min')
            del groupData
            aggData[vertex_begin:vertex_end] = aggRes
            del aggRes

        dist.all_reduce(aggData, op=dist.ReduceOp.MIN)
        actMask = aggData < vData
        vData = torch.min(aggData, vData)
        del aggData
        iter_num += 1

    t2 = time.time()
    maxVal = torch.max(vData[vData < INF]).item()
    nonVisited = torch.sum(vData >= INF).item()
    if rank == 0:
        per_iter = (t2 - t1) / iter_num if iter_num > 0 else 0
        print(f'Rank {rank} maxVal {int(maxVal)} nonVisitedNum {nonVisited} '
              f'iters {iter_num} one-iter time {per_iter}')


def bfs_bc(rank, world_size):
    """
    bfs_bc
    """
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # get paras
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str)
    parser.add_argument('--partition_num', '-n', type=int)
    parser.add_argument('--source', '-s', type=int, default=0)
    parser.add_argument('--long', '-l', type=int, default=0)
    args = parser.parse_args()

    data_path = args.data_path
    partition_num = args.partition_num
    source = args.source
    is_long = args.long

    # print('data_path: ', data_path, 'partition_num: ', partition_num, 'source: ', source)
    # init process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    

    local_rank = rank % torch.cuda.device_count()
    device = f'cuda:{local_rank}'

    # 分布式读图 + EBP 分区（rank 0 读图，broadcast row_ptr + scatter 分区）
    row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts = \
        distributed_read_and_partition(data_path, partition_num, is_long=is_long)

    INF = 10000

    # init data
    vData = torch.zeros(n_verts, dtype=torch.float32, device=device) + INF
    vData[source] = 0
    actMask = torch.zeros(n_verts, dtype=torch.int32, device=device)
    actMask[source] = 1
    modeSwitchThreshold = len(part_columns) // 15
    part_row_ptr = part_row_ptr.to(device)
    part_columns = part_columns.to(device)
    part_degrees = torch.diff(part_row_ptr)
    iter = 0
    aggData = torch.zeros_like(vData) # 用来保存每个subgraph的本地计算结果

    t1 = time.time()
    numAllreduce = 0
    while torch.any(actMask):
        aggData[:] = vData[:] # 保存上一轮的结果
        # print('aggData: ', aggData[0:40])
        while True:
            inner_iter = 0
            frontierLocal = torch.nonzero(actMask[vertex_begin : vertex_end]).view(-1)
            
            # print('rank {} iter {} inner_iter {} num_frontier {}'.format(rank, iter, inner_iter, len(frontierLocal)))
            actMask[vertex_begin : vertex_end] = False

            if len(frontierLocal) == 0:
                break
            numEdgesToProcess = torch.sum(part_degrees[frontierLocal])

            if numEdgesToProcess < modeSwitchThreshold:
                # print('frontierLocal: ', frontierLocal)
                starts = part_row_ptr[frontierLocal]
                ends = part_row_ptr[frontierLocal + 1]
                neighborIndices = multi_arange(starts, ends)
                neighbors = torch.index_select(part_columns, 0, neighborIndices)
                neighbors = neighbors.to(torch.int64)
                # ⚠ frontierLocal 是本地相对索引(0..n_local-1)，索引全局 aggData 需加 vertex_begin
                frontierLocalGlobal = frontierLocal + vertex_begin
                groupData = (torch.index_select(aggData, 0, frontierLocalGlobal) + 1).repeat_interleave(part_degrees[frontierLocal])
                aggRes = scatter(groupData, neighbors, dim_size=len(aggData), reduce='min')
                mask = aggRes > 0
                updateMask = mask & (aggRes < aggData)
                actMask = updateMask
                aggData = torch.where(updateMask, aggRes, aggData)
                del groupData, neighbors, aggRes
                # aggData = torch.min(aggData, aggRes)
                # print('rank {} iter {} : {} For Push Mode'.format(rank, iter, inner_iter))
            else:
                groupData =  torch.index_select(aggData, 0, part_columns)
                groupData.add_(1)
                groupID = part_row_ptr
                # segment_csr 要求 ptr 为 int64（PyTorch 2.x + torch_scatter 兼容）
                aggRes = segment_csr(groupData, groupID.to(torch.int64), reduce='min')
                updateMask = aggRes < aggData[vertex_begin : vertex_end]
                # ⚠ 修复 PyTorch 链式 fancy indexing 赋值不传播的 bug
                # 原：actMask[vertex_begin:vertex_end][updateMask] = True  ← 不生效
                # 原：aggData[vertex_begin:vertex_end][updateMask] = aggRes[updateMask]  ← 不生效
                # 正：用切片视图先取出 local_act/local_aggData 再 in-place 改，或用 where 整体替换
                local_act = actMask[vertex_begin : vertex_end]
                local_act[updateMask] = True
                local_agg = aggData[vertex_begin : vertex_end]
                local_agg[updateMask] = aggRes[updateMask]
                del groupData, groupID, aggRes
                # print('rank {} iter {} : {} For Pull Mode'.format(rank, iter, inner_iter))
            inner_iter += 1
        ta = time.time()
        dist.all_reduce(aggData, op=dist.ReduceOp.MIN) # min all reduce to get the global result
        tb = time.time()
        print(f'rank {rank} iter {iter} reduceTime {tb - ta}')
        numAllreduce += 1
        # print('data: ', aggData[0:40])
        actMask = aggData < vData
        vData = torch.min(aggData, vData)
        iter += 1
        print('rank {} iter {}'.format(rank, iter))
    t2 = time.time()
    mask = vData == INF
    nonVisitedNum = torch.sum(mask) 
    maxv = torch.max(vData[mask == 0])
    print('rank {} maxVal {} nonVisitedNum {} allReduceTime{} maxv{}'.format(rank, iter, nonVisitedNum, numAllreduce, maxv))
    print('vData: ', vData[0:40], 'elapsed time: ', t2 - t1)

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    mode = os.environ.get('BFS_MODE', 'bc')
    if mode == 'single':
        bfs_single(rank, world_size)
    else:
        bfs_bc(rank, world_size)