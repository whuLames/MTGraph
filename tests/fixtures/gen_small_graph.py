"""
生成一份合成 CSR 小图，用于回归测试。

输出（默认）：
  tests/fixtures/small/csr_vlist.bin   (int32)
  tests/fixtures/small/csr_elist.bin   (int32)

属性：
  - 100 个顶点、~500 条边
  - 顶点 ID 从 0 开始连续
  - 无自环、无重边
  - 弱连通（便于 BFS 跑出有意义结果）

用法：
  python tests/fixtures/gen_small_graph.py
  python tests/fixtures/gen_small_graph.py --vertices 200 --edges 1000 --out tests/fixtures/medium
  python tests/fixtures/gen_small_graph.py --preset medium      # 预设规模
"""
import argparse
import os
import random
import numpy as np


# 预设规模（vertices, edges）
PRESETS = {
    "tiny":   (20,   80),     # 极小，快速 CI
    "small":  (100,  500),    # 默认，回归测试
    "medium": (1000, 5000),   # 中等，benchmark
    "large":  (10000, 50000), # 较大，性能测试
}


def gen_csr(n_verts, n_edges, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    edges = set()
    # 先生成一条链保证弱连通
    for v in range(n_verts - 1):
        edges.add((v, v + 1))
        edges.add((v + 1, v))

    # 再随机补到 n_edges
    attempts = 0
    max_attempts = n_edges * 20
    while len(edges) < n_edges and attempts < max_attempts:
        u = random.randrange(n_verts)
        v = random.randrange(n_verts)
        attempts += 1
        if u == v:
            continue
        edges.add((u, v))

    # 按 src 聚合
    adj = [[] for _ in range(n_verts)]
    for u, v in edges:
        adj[u].append(v)
    for u in range(n_verts):
        adj[u].sort()

    # 构建 CSR
    row_ptr = np.zeros(n_verts + 1, dtype=np.int32)
    for u in range(n_verts):
        row_ptr[u + 1] = row_ptr[u] + len(adj[u])

    columns = []
    for u in range(n_verts):
        columns.extend(adj[u])
    columns = np.asarray(columns, dtype=np.int32)

    return row_ptr, columns, len(edges)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vertices", type=int, default=None)
    p.add_argument("--edges", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--preset", type=str, default=None,
                   choices=list(PRESETS.keys()),
                   help="使用预设规模（覆盖 --vertices/--edges/--out）")
    args = p.parse_args()

    # 处理 preset
    if args.preset:
        v, e = PRESETS[args.preset]
        args.vertices = args.vertices or v
        args.edges = args.edges or e
        if args.out is None:
            args.out = os.path.join(os.path.dirname(__file__), args.preset)

    # 默认值（无 preset 且无显式参数）
    if args.vertices is None:
        args.vertices = 100
    if args.edges is None:
        args.edges = 500
    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), "small")

    os.makedirs(args.out, exist_ok=True)
    row_ptr, columns, actual_edges = gen_csr(args.vertices, args.edges, args.seed)

    vlist_path = os.path.join(args.out, "csr_vlist.bin")
    elist_path = os.path.join(args.out, "csr_elist.bin")
    row_ptr.tofile(vlist_path)
    columns.tofile(elist_path)

    size_kb = (os.path.getsize(vlist_path) + os.path.getsize(elist_path)) / 1024.0
    print(f"✓ 生成图：{args.vertices} 顶点，{actual_edges} 边（目标 {args.edges}）")
    print(f"  {vlist_path}")
    print(f"  {elist_path}")
    print(f"  总大小: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
