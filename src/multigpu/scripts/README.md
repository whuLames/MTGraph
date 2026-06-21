# 多 GPU torch 脚本版（来自 TGraph/nonsingle/distributed/torch）

来自原始 TGraph 项目，是最干净的纯 torch 多 GPU 实现路径。
依赖：`torch` + `torch_scatter` + `torch.distributed`（NCCL 后端）。

## 文件

| 文件 | 算法 | 说明 |
|---|---|---|
| `pr.py` | PageRank | 边均衡划分 + 每轮 all_reduce(SUM) |
| `bfs.py` | BFS | 含两个变体：`bfs()` 单层 + `bfs_bc()` 本地多轮收敛（direction-optimizing）+ push/pull 自适应切换 |
| `run.py` | — | 进程组初始化 demo |
| `test.py` | — | `mp.Process` 启动 demo |
| `cmd.sh` | — | `source` 之，导出 MASTER_ADDR/MASTER_PORT |

## 用法

```bash
cd /path/to/MTGraphUnified
pip install -e .

# 1. 设置分布式环境（按需修改 MASTER_ADDR）
source src/multigpu/torch/scripts/cmd.sh

# 2. torchrun 启动（2 GPU）
torchrun --nproc_per_node=2 src/multigpu/torch/scripts/pr.py \
    --data_path /path/to/graph -n 2 -i 20

torchrun --nproc_per_node=2 src/multigpu/torch/scripts/bfs.py \
    --data_path /path/to/graph -n 2 -s 0
```

## 优化点（bfs.py 中的 `bfs_bc`）

- **push/pull 自适应切换**：`numEdgesToProcess < len(part_columns)//15` 时走 Push，否则 Pull。
- **本地多轮迭代**：内层 `while True` 在本地收敛后才做一次 `dist.all_reduce(MIN)`，显著减少通信次数。
- `multi_arange` / `read_data` / `get_partition` 复用自 `src.common.python`，无重复实现。
