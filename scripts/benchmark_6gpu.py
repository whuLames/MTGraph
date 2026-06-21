#!/usr/bin/env python
"""6-GPU benchmark: 4 algorithms × 4 datasets."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"
PORT = 42000

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

ALGOS = [
    ("pr",   "src/multigpu/scripts/pr.py",   {}),
    ("bfs_single", "src/multigpu/scripts/bfs.py", {"BFS_MODE": "single"}),
]

results = {}
for algo_name, script, extra_env in ALGOS:
    results[algo_name] = {}
    for ds_name, path, is_long in DATASETS:
        PORT += 1
        sys.stdout.write(f"  {algo_name} {ds_name} 6-GPU ... ")
        sys.stdout.flush()
        cmd = [
            TORCHRUN, "--nproc_per_node=6",
            "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
            os.path.join(PROJ, script),
            "--data_path", path, "-n", "6", "-l", is_long, "-i", "10", "-s", "0",
        ]
        env = dict(os.environ, **extra_env)
        timeout = 900 if ds_name in ("sk-2005", "uk-2007") else 300
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ, env=env)
            out = r.stdout + r.stderr
            val = None
            for line in out.splitlines():
                if "Rank 0" in line and "one-iter time" in line:
                    m = re.search(r"one-iter time ([\d.eE+-]+)", line)
                    if m: val = float(m.group(1))
            if val:
                results[algo_name][ds_name] = val
                print(f"{val*1000:.1f} ms/iter")
            else:
                results[algo_name][ds_name] = "FAIL"
                print("FAIL")
        except Exception as e:
            results[algo_name][ds_name] = "FAIL"
            print("FAIL")

# WCC
results["wcc"] = {}
for ds_name, path, is_long in DATASETS:
    PORT += 1
    sys.stdout.write(f"  wcc {ds_name} 6-GPU ... ")
    sys.stdout.flush()
    cmd = [TORCHRUN, "--nproc_per_node=6", "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
           os.path.join(PROJ, "src/multigpu/scripts/wcc.py"), "--data_path", path, "-n", "6", "-l", is_long, "-i", "50"]
    env = dict(os.environ)
    timeout = 900 if ds_name in ("sk-2005", "uk-2007") else 300
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ, env=env)
        out = r.stdout + r.stderr
        val = None
        for line in out.splitlines():
            if "Rank 0" in line and "one-iter time" in line:
                m = re.search(r"one-iter time ([\d.eE+-]+)", line)
                if m: val = float(m.group(1))
        if val:
            results["wcc"][ds_name] = val
            print(f"{val*1000:.1f} ms/iter")
        else:
            results["wcc"][ds_name] = "FAIL"
            print("FAIL")
    except:
        results["wcc"][ds_name] = "FAIL"
        print("FAIL")

# SSSP
results["sssp"] = {}
for ds_name, path, is_long in DATASETS:
    PORT += 1
    sys.stdout.write(f"  sssp {ds_name} 6-GPU ... ")
    sys.stdout.flush()
    cmd = [TORCHRUN, "--nproc_per_node=6", "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
           os.path.join(PROJ, "src/multigpu/scripts/sssp.py"), "--data_path", path, "-n", "6", "-l", is_long, "-i", "50", "-s", "0"]
    env = dict(os.environ)
    timeout = 900 if ds_name in ("sk-2005", "uk-2007") else 300
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ, env=env)
        out = r.stdout + r.stderr
        val = None
        for line in out.splitlines():
            if "Rank 0" in line and "one-iter time" in line:
                m = re.search(r"one-iter time ([\d.eE+-]+)", line)
                if m: val = float(m.group(1))
        if val:
            results["sssp"][ds_name] = val
            print(f"{val*1000:.1f} ms/iter")
        else:
            results["sssp"][ds_name] = "FAIL"
            print("FAIL")
    except:
        results["sssp"][ds_name] = "FAIL"
        print("FAIL")

# 输出表格
print("\n" + "=" * 60)
print("6-GPU Benchmark Results (ms/iter)")
print("=" * 60)
print(f"{'Dataset':<16} {'PR':>10} {'BFS-single':>12} {'WCC':>10} {'SSSP':>10}")
print("-" * 60)
for ds_name, _, _ in DATASETS:
    row = f"{ds_name:<16}"
    for algo in ["pr", "bfs_single", "wcc", "sssp"]:
        v = results[algo].get(ds_name)
        if isinstance(v, float):
            row += f" {v*1000:>9.1f}"
        else:
            row += f" {'FAIL':>10}"
    print(row)

# 保存 JSON
import json
with open(os.path.join(PROJ, "benchmark_6gpu.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to benchmark_6gpu.json")
