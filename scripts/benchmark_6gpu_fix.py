#!/usr/bin/env python
"""6-GPU PR + BFS benchmark (fix for PATH issue)."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"
PORT = 44000

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

# WCC+SSSP from previous run (already have results)
WCC_6GPU = {"soc-twitter": 65.3, "soc-sinaweibo": 135.4, "sk-2005": 330.7, "uk-2007": 108.6}
SSSP_6GPU = {"soc-twitter": 64.6, "soc-sinaweibo": 138.9, "sk-2005": 280.5, "uk-2007": 136.6}

def extract_per_iter(out):
    for line in out.splitlines():
        if "Rank 0" in line and "one-iter time" in line:
            m = re.search(r"one-iter time ([\d.eE+-]+)", line)
            if m: return float(m.group(1))
    return None

pr_results = {}
bfs_results = {}

for name, path, is_long in DATASETS:
    PORT += 1
    timeout = 900 if name in ("sk-2005", "uk-2007") else 300

    # PR (no -s flag)
    sys.stdout.write(f"  PR {name} 6-GPU ... "); sys.stdout.flush()
    cmd = [TORCHRUN, "--nproc_per_node=6", "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
           os.path.join(PROJ, "src/multigpu/scripts/pr.py"), "--data_path", path, "-n", "6", "-i", "10", "-l", is_long]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ)
        v = extract_per_iter(r.stdout + r.stderr)
        pr_results[name] = v
        print(f"{v*1000:.1f} ms" if v else "FAIL")
    except: pr_results[name] = None; print("FAIL")

    # BFS single (with -s 0, BFS_MODE=single)
    PORT += 1
    sys.stdout.write(f"  BFS {name} 6-GPU ... "); sys.stdout.flush()
    cmd = [TORCHRUN, "--nproc_per_node=6", "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
           os.path.join(PROJ, "src/multigpu/scripts/bfs.py"), "--data_path", path, "-n", "6", "-s", "0", "-l", is_long]
    env = dict(os.environ, BFS_MODE="single")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ, env=env)
        v = extract_per_iter(r.stdout + r.stderr)
        bfs_results[name] = v
        print(f"{v*1000:.1f} ms" if v else "FAIL")
    except: bfs_results[name] = None; print("FAIL")

# 完整 6-GPU 表格
print("\n" + "=" * 70)
print("6-GPU Complete Benchmark (ms/iter)")
print("=" * 70)
print(f"{'Dataset':<16} {'PR':>8} {'BFS':>8} {'WCC':>8} {'SSSP':>8}")
print("-" * 70)
for name, _, _ in DATASETS:
    pr = pr_results.get(name)
    bfs = bfs_results.get(name)
    wcc = WCC_6GPU.get(name)
    sssp = SSSP_6GPU.get(name)
    pr_s = f"{pr*1000:.1f}" if pr else "FAIL"
    bfs_s = f"{bfs*1000:.1f}" if bfs else "FAIL"
    wcc_s = f"{wcc:.1f}" if wcc else "FAIL"
    sssp_s = f"{sssp:.1f}" if sssp else "FAIL"
    print(f"{name:<16} {pr_s:>8} {bfs_s:>8} {wcc_s:>8} {sssp_s:>8}")

# 保存结果
import json
all_6gpu = {"PR": pr_results, "BFS_single": bfs_results, "WCC": WCC_6GPU, "SSSP": SSSP_6GPU}
with open(os.path.join(PROJ, "benchmark_6gpu.json"), "w") as f:
    json.dump(all_6gpu, f, indent=2, default=str)
