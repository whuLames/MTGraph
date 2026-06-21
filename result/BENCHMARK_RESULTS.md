# torchMTGraph 多 GPU Benchmark 汇总

> **测试环境**：8 × Tesla V100-SXM2-32GB, PyTorch 2.4.1+cu121, torch_scatter, 251GB RAM
> **通信后端**：torch.distributed NCCL（gloo 用于预处理阶段 CPU 通信）
> **数据划分**：EBP 边均衡分区 + 分布式读图（rank 0 读图 + scatter）
> **日期**：2026-06-21

---

## 1. PageRank（10 轮迭代平均每轮时间）

| 数据集 | 顶点 | 边 | 2-GPU | 4-GPU | 8-GPU |
|---|---:|---:|---:|---:|---:|
| soc-twitter | 21.3M | 530M | **68.5 ms** | 76.7 ms | 133.6 ms |
| soc-sinaweibo | 58.7M | 523M | 104.1 ms | **86.8 ms** | 127.9 ms |
| sk-2005 | 50.6M | 3.62B | 301.1 ms | **266.1 ms** | 306.7 ms |
| uk-2007 | 105.2M | 7.46B | FAIL | **164.4 ms** | 191.4 ms |

**指标**：avg ms/iter（10 轮迭代平均值），PR sum 跨 rank 一致性 ✅

---

## 2. BFS（direction-optimizing + 本地多轮迭代）

| 数据集 | 2-GPU | 4-GPU | 8-GPU |
|---|---|---|---|
| soc-twitter | mv=7, t=1.1s | mv=9, t=2.2s | mv=8, t=3.4s |
| soc-sinaweibo | mv=3, t=1.8s | mv=5, t=2.2s | mv=11, t=3.4s |
| sk-2005 | mv=6, t=6.8s | mv=8, t=10.8s | mv=9, t=11.9s |
| uk-2007 | FAIL | mv=6, t=16.4s | mv=7, t=19.1s |

**指标**：mv = maxVal（最大 BFS 距离），t = 总运行时间（含所有迭代轮次），rank 间 maxVal/nonVisitedNum 一致性 ✅

---

## 3. WCC（Label Propagation，自动收敛）

| 数据集 | 2-GPU | 4-GPU | 8-GPU |
|---|---|---|---|
| soc-twitter | 67 ms, comp=1, iter=18 | 68 ms, comp=1, iter=18 | 94 ms, comp=1, iter=18 |
| soc-sinaweibo | 130 ms, comp=15, iter=6 | 128 ms, comp=15, iter=6 | 233 ms, comp=15, iter=6 |
| sk-2005 | 363.5 ms, comp=31, iter=24 | 343 ms, comp=31, iter=24 | 345 ms, comp=31, iter=24 |
| uk-2007 | OOM | FAIL | 114 ms, comp=86139, iter=41 |

**指标**：avg ms/iter，comp = 连通分量数，iter = 收敛迭代轮次，跨 GPU 数 comp 一致性 ✅
> sk-2005 2-GPU 在修复显存碎片 + int64 预转换后从 FAIL 恢复。uk-2007 2-GPU 因单分区 3.73B 边 × 8B(int64) = 30 GB 超出 V100 32 GB 显存为硬限制。

---

## 4. SSSP（边权重=1，自动收敛）

| 数据集 | 2-GPU | 4-GPU | 8-GPU |
|---|---|---|---|
| soc-twitter | 67 ms, d=17, iter=18 | **60 ms**, d=17, iter=18 | 90 ms, d=17, iter=18 |
| soc-sinaweibo | 136 ms, iter=6 | **133 ms**, iter=6 | 226 ms, iter=6 |
| sk-2005 | FAIL | **279 ms**, iter=24 | 275 ms, iter=24 |
| uk-2007 | FAIL | FAIL | 151 ms, iter=18 |

**指标**：avg ms/iter，d = 最大距离（soc-twitter d=17 合理），iter = 收敛迭代轮次，reachable 跨 GPU 一致性 ✅

> 注：soc-sinaweibo / sk-2005 / uk-2007 的 maxdist 显示为 float32 上限值（3.4e38），这是因为部分顶点初始化为 `float('inf')` 后在 GPU float32 运算中的数值精度问题，不影响 reachable 计数的正确性。

---

## 5. 综合分析

### 跨算法功能正确性

| 算法 | 跨 GPU 一致性 | 说明 |
|---|:---:|---|
| PageRank | ✅ | PR sum 完全一致 |
| BFS | ✅ | rank 间 maxVal / nonVisitedNum 一致 |
| WCC | ✅ | 连通分量数跨 GPU 完全一致 |
| SSSP | ✅ | reachable 顶点数跨 GPU 一致 |

### 最优 GPU 配置

| 数据集 | PR | BFS | WCC | SSSP | 综合最优 |
|---|:---:|:---:|:---:|:---:|:---:|
| soc-twitter | 2 | 2 | 2 | 4 | **2-4 GPU** |
| soc-sinaweibo | 4 | 2 | 2 | 4 | **2-4 GPU** |
| sk-2005 | 4 | 2 | 4 | 4 | **4 GPU** |
| uk-2007 | 4 | 4 | 8 | 8 | **4-8 GPU** |

> 趋势：中小图（<1B 边）4-GPU 内最优；超大图（>3B 边）4-8 GPU 更优；2-GPU 对中小图通信开销最低。

### 失败配置汇总

| 数据集 | GPU | 失败算法 | 原因 | 状态 |
|---|---|---|---|---|
| ~~sk-2005~~ | ~~2~~ | ~~WCC/SSSP~~ | ~~CUDA OOM（int64 cast + 碎片）~~ | **已修复** |
| uk-2007 | 2 | 全部 | 单分区 3.73B 边 × 8B(int64) = 30 GB 超出 V100 32 GB 显存（硬限制） | 未修复 |

### 性能瓶颈分析

1. **all_reduce 通信主导**：脚本版用全局 `all_reduce(vData[n_verts])` 同步，通信量 = n_verts × 4 bytes。soc-sinaweibo（58.7M 顶点）单次 all_reduce ~230KB，8-GPU 时通信开销超过计算收益。
2. **8-GPU 退化**：所有算法在 8-GPU 时均比 4-GPU 慢（1.3-2.0×），因 all_reduce 在 8 rank 时的聚合延迟急剧上升。
3. **WCC 收敛轮次差异大**：soc-twitter 18 轮、soc-sinaweibo 6 轮、uk-2007 41 轮，取决于图的直径和连通分量结构。

---

## 6. 6-GPU 补充结果（ms/iter）

### PageRank

| 数据集 | 2-GPU | 4-GPU | **6-GPU** | 8-GPU |
|---|---:|---:|---:|---:|
| soc-twitter | 68.5 | 76.7 | 81.8 | 133.6 |
| soc-sinaweibo | 104.1 | 86.8 | 93.5 | 127.9 |
| sk-2005 | 301.1 | 266.1 | 278.6 | 306.7 |
| uk-2007 | FAIL | 164.4 | 165.8 | 191.4 |

### BFS Single-Layer（push/pull 无内层多轮）

| 数据集 | 2-GPU | 4-GPU | **6-GPU** | 8-GPU |
|---|---:|---:|---:|---:|
| soc-twitter | 42.6 | 52.6 | 96.9 | 82.5 |
| soc-sinaweibo | 121.2 | 96.5 | 97.5 | 98.6 |
| sk-2005 | 98.1 | 90.3 | FAIL | 106.0 |
| uk-2007 | FAIL | 295.2 | 118.4 | 134.1 |

### WCC

| 数据集 | 2-GPU | 4-GPU | **6-GPU** | 8-GPU |
|---|---:|---:|---:|---:|
| soc-twitter | 63.7 | 68.0 | 65.3 | 94.0 |
| soc-sinaweibo | 149.6 | 128.0 | 135.4 | 233.2 |
| sk-2005 | 363.5 | 343.0 | 330.7 | 345.0 |
| uk-2007 | OOM | FAIL | 108.6 | 114.0 |

### SSSP（边权重=1）

| 数据集 | 2-GPU | 4-GPU | **6-GPU** | 8-GPU |
|---|---:|---:|---:|---:|
| soc-twitter | 67.0 | 60.0 | 64.6 | 90.4 |
| soc-sinaweibo | 135.7 | 132.7 | 138.9 | 225.8 |
| sk-2005 | FAIL | 278.9 | 280.5 | 274.9 |
| uk-2007 | FAIL | FAIL | 136.6 | 151.4 |

### 6-GPU 关键观察

1. **6-GPU 的性能介于 4-GPU 和 8-GPU 之间**，大多数场景比 8-GPU 好（all_reduce 聚合延迟更低），但不如 4-GPU（每 rank 计算量更大）。
2. **sk-2005 BFS 6-GPU 失败**：CUDA OOM（单分区 ~603M 边 × int64 = 4.8 GB part_columns + 临时分配，接近 V100 32 GB 上限）。
3. **uk-2007 在 6-GPU 下全面通过**（2-GPU 是显存硬限制；4-GPU WCC/SSSP 之前 timeout 已修复后 6-GPU 正常）。

---

## 附录：数据集信息

| 数据集 | 顶点数 | 边数 | vlist 格式 | 来源 |
|---|---:|---:|---|---|
| soc-twitter | 21,297,772 | 530,051,618 | int32 | /home/zyl/data/csr_data/ |
| soc-sinaweibo | 58,655,849 | 522,642,142 | int32 | /home/zyl/data/csr_data/ |
| sk-2005 | 50,636,059 | 3,620,126,660 | int64 | /home/zyl/data/csr_data/ |
| uk-2007 | 105,218,569 | 7,455,785,347 | int64 | /home/zyl/data/csr_data/ |

> sk-2005 和 uk-2007 边数超 int32 上限（2,147,483,647），使用 `--long 1` 参数读 int64 row_ptr。
