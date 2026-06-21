# torchMTGraph

纯 torch 实现的多 GPU 图计算框架，整合自 TGraph（GASProgram 框架）与 MTGraph（公共模块）。

## 项目特点

- **纯 PyTorch 实现**：无 CUDA C++ 代码，仅依赖 `torch` + `torch_scatter` + `numpy`
- **GASProgram 框架抽象**：仿 PowerGraph 的 Gather-Sum-Apply-Scatter 模型，算法通过继承实现
- **两种多 GPU 路线**：
  - `scripts/`：最纯 torch 脚本版（含 push/pull + 本地多轮迭代优化）
  - `framework/`：基于 GASProgram + MultiGPUStrategyByNCCL 的框架版
- **5 个基础算法**：BFS、PageRank、SSSP、WCC、HITS

## 目录结构

```
torchMTGraph/
├── src/
│   ├── common/python/        # 公共模块（io / partition / arange / comm_plan / distributed）
│   ├── type/                 # 图类型系统（Graph / CSRGraph / CSRCGraph / CSCGraph / Subgraph）
│   ├── framework/            # ⭐ GASProgram 框架
│   │   ├── GASProgram.py            # GAS 抽象基类
│   │   ├── helper.py
│   │   ├── partition/               # EdgePartition / VertexPartition / GeminiPartition
│   │   └── strategy/                # SimpleStrategy / PartitionStrategy / MultiGPUStrategyByNCCL 等
│   ├── algorithms/           # 5 个算法的 GASProgram 子类
│   │   ├── BFS.py / PageRank.py / ShortestPaths.py / ConnectedComponents.py / HITS.py
│   ├── multigpu/             # ⭐ 多 GPU 实现
│   │   ├── scripts/                  # 脚本版（最纯 torch，独立无框架依赖）
│   │   │   ├── bfs.py                   # 含 push/pull + 本地多轮迭代（bfs_bc）
│   │   │   └── pr.py
│   │   └── framework/               # 框架版（GASProgram + NCCL strategy）
│   │       └── PageRank.py
│   └── apps/                 # 单 GPU 算法入口（基于 SimpleStrategy）
├── tests/                    # pytest 单元测试 + fixtures
├── tools/                    # 图预处理（el2csr.cpp）
├── scripts/                  # 启动辅助（env.sh / cmd.sh）
├── pyproject.toml
└── Makefile
```

## 快速上手

### 1. 准备 conda 环境（推荐 PyTorch 2.4.1，与 torch_scatter 兼容）

```bash
conda create -n torch_mtgraph python=3.10 -y
conda activate torch_mtgraph
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install pytest numpy
```

### 2. 安装项目（在 torchMTGraph 根目录）

```bash
cd /path/to/torchMTGraph
pip install -e .
```

### 3. 单 GPU 算法入口

```bash
# BFS（fixtures 小图）
python -m src.apps.bfs --graph tests/fixtures/small --output /tmp/bfs.out --source 0 --cuda

# PageRank
python -m src.apps.pagerank --graph tests/fixtures/small --output /tmp/pr.out --cuda
```

### 4. 多 GPU 脚本版（torchrun）

```bash
source scripts/env.sh

# PageRank 2 GPU
torchrun --nproc_per_node=2 src/multigpu/scripts/pr.py \
    --data_path tests/fixtures/small --partition_num 2 --iterations 20

# BFS 2 GPU（默认 bfs_bc 变体，含本地多轮迭代）
torchrun --nproc_per_node=2 src/multigpu/scripts/bfs.py \
    --data_path tests/fixtures/small --partition_num 2 --source 0
```

### 5. 多 GPU 框架版（GASProgram + Strategy）

```bash
# PageRank 多 GPU（内部用 mp.spawn 启动 N 进程，每进程绑定 1 GPU）
python src/multigpu/framework/PageRank.py \
    --graph tests/fixtures/small --device_num 2 --num_iter 20
```

### 6. 单元测试

```bash
pytest tests/ -v
```

## 关键设计

### GASProgram 抽象

所有算法继承 `GASProgram`，实现 4 个抽象方法：

| 方法 | 语义 |
|---|---|
| `gather(vertices, nbrs, edges, ptr)` | 从邻居收集信息 |
| `sum(gathered_data, ptr)` | 对收集的信息做归约（sum/min/max） |
| `apply(vertices, gathered_sum)` | 把归约结果应用到顶点 |
| `scatter(vertices, nbrs, edges, ptr, apply_data)` | 把更新传播给邻居 |

运行时通过注入 `Strategy` 决定执行后端（单 GPU / 多 GPU）。

### 多 GPU 两条路线

| 路线 | 入口 | 抽象层 | 算法级优化 |
|---|---|---|---|
| `multigpu/scripts/` | 独立脚本 | 无 | BFS 含 push/pull + 本地多轮迭代 |
| `multigpu/framework/` | GASProgram 子类 | Strategy 模式 | 无（朴素 GAS 流程） |

两者通信都基于 `torch.distributed` NCCL 后端，用 `all_reduce` 同步全局顶点数据。

### 数据格式

CSR 二进制：`<path>/csr_vlist.bin`（行指针）+ `<path>/csr_elist.bin`（列索引）。
`CSRCGraph.read_csrc_graph_bin` 自动从 CSR 构造 CSC（如果文件缺失）。

预处理：`g++ tools/el2csr.cpp -o el2csr && ./el2csr input.el output_dir/`

## 算法覆盖

| 算法 | 单 GPU 入口 | 多 GPU 脚本版 | 多 GPU 框架版 |
|---|---|---|---|
| BFS | `apps/bfs.py` | `multigpu/scripts/bfs.py` | 待扩展 |
| PageRank | `apps/pagerank.py` | `multigpu/scripts/pr.py` | `multigpu/framework/PageRank.py` |
| SSSP | `apps/sssp.py` | 待扩展 | 待扩展 |
| WCC | `apps/wcc.py` | 待扩展 | 待扩展 |
| HITS | `apps/hits.py` | 待扩展 | 待扩展 |

## 已知限制

- 多 GPU 框架版当前仅 PageRank demo；BFS/SSSP/WCC/HITS 需参照 PageRank 模式扩展
- 多 GPU 脚本版仅 BFS + PageRank（BFS 含 push/pull 自适应 + 本地多轮迭代优化）
- 依赖 `torch_scatter`（PyTorch 2.8 暂无对应预编译 wheel，建议用 PyTorch 2.4.x）

## 性能 profiling（viztracer，可选）

多 GPU 框架版的 `MultiGPUStrategyByNCCL` 内置 [viztracer](https://github.com/gaogaotiantian/viztracer) 钩子，**默认关闭**。开启方式：

```bash
# 安装 viztracer（一次性）
pip install viztracer

# 启用 profile
MTGRAPH_PROFILE=1 python src/multigpu/framework/PageRank.py \
    --graph tests/fixtures/small --device_num 2 --num_iter 20

# 输出：每 rank 一个 trace 文件（result0.json、result1.json ...）
# 查看：
#   - 在线：把 json 上传到 https://ui.perfetto.dev/
#   - 本地：vizviewer result0.json
```

不设环境变量时跳过 profile（零开销）。trace 文件已加入 `.gitignore`，不会误提交。
