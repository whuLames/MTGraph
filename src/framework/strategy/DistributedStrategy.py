"""
DistributedStrategy：统一的多 GPU 迭代调度框架（支持 push/pull 自适应切换）。

设计原则：
  - 算法层（GASProgram 子类）零修改
  - 所有分布式语义（读图、分区、通信、收敛、push/pull 切换）在本 Strategy 中
  - 算法特定的计算逻辑通过回调函数注入

两种计算模式：
  - Pull（默认）：每顶点拉取所有邻居做归约（segment_csr），计算量与 frontier 无关
  - Push（可选）：frontier 顶点主动把更新推向邻居（scatter），计算量与 frontier 成正比

切换由 frontier 边数阈值决定（direction-optimizing）。

用法示例（BFS with push/pull）：
    strategy = DistributedStrategy(
        data_path=path, partition_num=4,
        reduce='min', allreduce_op=dist.ReduceOp.MIN,
        init_fn=..., gather_fn=..., apply_fn=...,
        # push/pull 配置
        enable_push_pull=True,
        push_threshold_ratio=1/15,
        frontier_fn=lambda vd, vb, ve, ctx: torch.nonzero(vd[vb:ve] < ctx['INF']).view(-1),
        push_value_fn=lambda vd, frontier_global, ctx: vd[frontier_global] + 1.0,
        max_iters=50, fixed_iters=False,
    )
"""
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch_scatter import segment_csr, scatter
import time

from src.framework.strategy.Strategy import Strategy
from src.common.python.distributed import distributed_read_and_partition
from src.common.python.arange import multi_arange


class DistributedStrategy(Strategy):
    """
    多 GPU 迭代调度框架。统一处理 PR / BFS / WCC / SSSP 的多 GPU 计算。

    Pull-only 用法（PR / WCC / SSSP 等）：
        只需提供 init_fn / gather_fn / apply_fn。

    Push+Pull 用法（BFS 等 direction-optimizing 场景）：
        额外提供 enable_push_pull=True + frontier_fn + push_value_fn。
        Strategy 自动根据 frontier 大小在 push/pull 之间切换。

    Args:
        data_path:        图数据目录
        partition_num:    分区数（= world_size）
        reduce:           segment_csr / scatter 的 reduce 模式（'sum' / 'min'）
        allreduce_op:     dist.all_reduce 的 op（dist.ReduceOp.SUM / MIN）
        init_fn:          vData 初始化函数 (n_verts, device) -> Tensor
        gather_fn:        pull 模式 gather 变换 (vData, columns, ctx) -> Tensor
        apply_fn:         本地更新函数 (old_local, aggRes, ctx) -> Tensor
        prepare_fn:       可选预处理 (row_ptr, device) -> dict（算法特有的常量）
        max_iters:        最大迭代轮次
        fixed_iters:      True=固定轮次（PR），False=自动收敛（BFS/WCC/SSSP）
        is_long:          0=int32 row_ptr, 1=int64（边数 > 2^31 时用）
        source:           BFS/SSSP 的源点

    Push/Pull 相关参数（可选）：
        enable_push_pull:       是否启用 push/pull 自适应切换（默认 False=纯 pull）
        push_threshold_ratio:   frontier 边数 / 本地总边数的切换阈值（默认 1/15）
        frontier_fn:            找本地活跃顶点 (vData, vb, ve, ctx) -> Tensor（本地相对索引）
        push_value_fn:          计算 push 值 (vData, frontier_global, ctx) -> Tensor
    """

    def __init__(self, data_path, partition_num,
                 reduce='min', allreduce_op=dist.ReduceOp.MIN,
                 init_fn=None, gather_fn=None, apply_fn=None, prepare_fn=None,
                 max_iters=50, fixed_iters=False, is_long=0, source=0,
                 # push/pull 参数
                 enable_push_pull=False, push_threshold_ratio=1.0/15,
                 frontier_fn=None, push_value_fn=None):
        super().__init__()
        self.data_path = data_path
        self.partition_num = partition_num
        self.reduce = reduce
        self.allreduce_op = allreduce_op
        self.init_fn = init_fn
        self.gather_fn = gather_fn
        self.apply_fn = apply_fn
        self.prepare_fn = prepare_fn
        self.max_iters = max_iters
        self.fixed_iters = fixed_iters
        self.is_long = is_long
        self.source = source
        # push/pull
        self.enable_push_pull = enable_push_pull
        self.push_threshold_ratio = push_threshold_ratio
        self.frontier_fn = frontier_fn
        self.push_value_fn = push_value_fn

    def _do_pull(self, vData, part_row_ptr, part_columns, vertex_begin, vertex_end, ctx, device):
        """Pull 模式：每顶点拉取所有邻居做归约。"""
        aggData = self.gather_fn(vData, part_columns, ctx)
        aggRes = segment_csr(aggData, part_row_ptr, reduce=self.reduce)
        del aggData
        local_old = vData[vertex_begin:vertex_end]
        local_new = self.apply_fn(local_old, aggRes, ctx)
        changed = (local_new != local_old).sum().item()
        vData[vertex_begin:vertex_end] = local_new
        del aggRes, local_old, local_new
        return changed

    def _do_push(self, vData, part_row_ptr, part_columns, part_degrees,
                 vertex_begin, vertex_end, ctx, device, n_verts,
                 frontier_local, frontier_global):
        """Push 模式：frontier 顶点主动推向邻居。

        frontier_local / frontier_global 由 compute() 从 act_mask 计算后传入。
        """
        if len(frontier_local) == 0:
            return 0

        # 展开活跃顶点的出边
        starts = part_row_ptr[frontier_local]
        ends = part_row_ptr[frontier_local + 1]
        edge_indices = multi_arange(starts, ends)

        # 计算推送值 + 目标邻居
        neighbors = part_columns[edge_indices]
        push_vals = self.push_value_fn(vData, frontier_global, ctx)
        push_vals = push_vals.repeat_interleave(part_degrees[frontier_local])
        del starts, ends

        # scatter 到全局邻居
        aggRes = scatter(push_vals, neighbors, dim_size=n_verts, reduce=self.reduce)
        del push_vals, neighbors, edge_indices

        # 只更新被 push 到的本地顶点
        if self.reduce == 'min':
            mask = aggRes > 0
            local_old = vData[vertex_begin:vertex_end]
            local_new = torch.where(mask[vertex_begin:vertex_end],
                                    torch.min(local_old, aggRes[vertex_begin:vertex_end]),
                                    local_old)
        else:
            local_old = vData[vertex_begin:vertex_end]
            local_new = local_old + aggRes[vertex_begin:vertex_end]

        changed = (local_new != local_old).sum().item()
        vData[vertex_begin:vertex_end] = local_new
        del aggRes, local_old, local_new
        return changed

    def compute(self):
        """执行多 GPU 迭代计算。返回 (vData, stats)。"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f'cuda:{rank % torch.cuda.device_count()}'

        # ---- 1. 分布式读图 + EBP 分区 ----
        row_ptr, part_row_ptr, part_columns, vertex_begin, vertex_end, n_verts = \
            distributed_read_and_partition(
                self.data_path, self.partition_num, is_long=self.is_long)

        part_row_ptr = part_row_ptr.to(device).to(torch.int64)
        part_columns = part_columns.to(device).to(torch.int64)

        if rank == 0:
            mode_str = 'push+pull' if self.enable_push_pull else 'pull-only'
            print(f'[DistributedStrategy] v={n_verts} mode={mode_str} distributed read done')

        # ---- 2. 算法特有的预处理 ----
        ctx = {}
        if self.prepare_fn is not None:
            ctx = self.prepare_fn(row_ptr, device)

        # ---- 3. 初始化 vData ----
        vData = self.init_fn(n_verts, device)

        # act_mask: 跟踪"本轮有更新的顶点"（push 模式的 frontier 来源）
        act_mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
        if hasattr(self, 'source') and self.source is not None:
            act_mask[self.source] = True

        # push/pull 切换阈值
        part_degrees = None
        push_threshold = None
        if self.enable_push_pull:
            part_degrees = torch.diff(part_row_ptr)
            push_threshold = int(len(part_columns) * self.push_threshold_ratio)

        # ---- 4. 迭代 ----
        t1 = time.time()
        actual_iters = 0
        push_count = 0
        pull_count = 0

        for i in range(self.max_iters):
            actual_iters = i + 1

            if self.enable_push_pull:
                # frontier 来自 act_mask（本轮有更新的顶点）
                frontier_local = torch.nonzero(act_mask[vertex_begin:vertex_end]).view(-1)
                num_edges = torch.sum(part_degrees[frontier_local]).item() if len(frontier_local) > 0 else 0

                if num_edges < push_threshold and len(frontier_local) > 0:
                    # ---- Push 模式 ----
                    frontier_global = frontier_local + vertex_begin
                    changed = self._do_push(vData, part_row_ptr, part_columns,
                                            part_degrees, vertex_begin, vertex_end,
                                            ctx, device, n_verts, frontier_local, frontier_global)
                    del frontier_global
                    push_count += 1
                else:
                    # ---- Pull 模式 ----
                    changed = self._do_pull(vData, part_row_ptr, part_columns,
                                           vertex_begin, vertex_end, ctx, device)
                    pull_count += 1
                del frontier_local
            else:
                # ---- 纯 Pull 模式 ----
                changed = self._do_pull(vData, part_row_ptr, part_columns,
                                       vertex_begin, vertex_end, ctx, device)
                pull_count += 1

            # 记录 all_reduce 前的 vData（用于更新 act_mask）
            prev_vData = vData.clone()

            # all_reduce: 全局同步
            dist.all_reduce(vData, op=self.allreduce_op)

            # 更新 act_mask：本轮全局有变化的顶点
            act_mask = (vData != prev_vData)
            del prev_vData

            # 收敛检查
            if not self.fixed_iters:
                changed_tensor = torch.tensor([changed], device=device, dtype=torch.int64)
                dist.all_reduce(changed_tensor, op=dist.ReduceOp.SUM)
                if changed_tensor.item() == 0:
                    if rank == 0:
                        print(f'[DistributedStrategy] converged at iter {actual_iters} '
                              f'(push={push_count}, pull={pull_count})')
                    break

        t2 = time.time()
        per_iter = (t2 - t1) / actual_iters if actual_iters > 0 else 0

        if rank == 0:
            print(f'[DistributedStrategy] Rank {rank} iters {actual_iters} '
                  f'push={push_count} pull={pull_count} '
                  f'one-iter time {per_iter}')

        return vData, {'iters': actual_iters, 'per_iter': per_iter,
                       'push_count': push_count, 'pull_count': pull_count}
