# 多 GPU 框架版（GASProgram + MultiGPUStrategyByNCCL）

基于 GASProgram 抽象，通过切换 `Strategy` 实现多 GPU 计算。当前包含 PageRank 一个 demo。

## 文件

| 文件 | 算法 | 说明 |
|---|---|---|
| `PageRank.py` | PageRank | 基于 `MultiGPUStrategyByNCCL`（torch.distributed NCCL） |

## 用法

```bash
# 单机多 GPU 启动（2 GPU）
python src/multigpu/framework/PageRank.py \
    --graph tests/fixtures/small --device_num 2 --num_iter 20
```

`MultiGPUStrategyByNCCL` 内部用 `torch.multiprocessing.spawn` 启动 `device_num` 个子进程，每个进程绑定一个 GPU，通过 `torch.distributed` NCCL 后端做 all-reduce 同步。

## 与脚本版的对比

| 维度 | `scripts/`（脚本版） | `framework/`（框架版） |
|---|---|---|
| 抽象层 | 无（独立脚本） | GASProgram（gather/sum/apply/scatter） |
| 算法覆盖 | BFS + PageRank | PageRank（BFS/SSSP/WCC/HITS 可仿照扩展） |
| 算法级优化 | 含 push/pull + 本地多轮迭代（BFS） | 无（朴素 GAS 流程） |
| 通信 | torch.distributed NCCL all_reduce(SUM) | 同左 |
| 复用性 | 低（每算法独立脚本） | 高（继承 GASProgram 即可） |

## 扩展其他算法到框架版

参照 `PageRank.py` 的模式：
1. 在 `src/algorithms/` 找到对应算法的 GASProgram 子类（如 `BFS`）；
2. 复用算法类，切换 strategy 为 `MultiGPUStrategyByNCCL`；
3. 启动入口写 main 函数。

示例（假设要加 BFS 框架版）：
```python
from src.algorithms.BFS import BFS
from src.framework.strategy.MultiGPUStrategyByNCCL import MultiGPUStrategyByNCCL
from src.framework.partition.GeminiPartition import GeminiPartition

bfs = BFS(graph, start_from=[0])
partition = GeminiPartition(graph, num_partitions=N, alpha=8*N-8)
strategy = MultiGPUStrategyByNCCL(bfs, partition, device_num=N)
strategy.compute()
```
