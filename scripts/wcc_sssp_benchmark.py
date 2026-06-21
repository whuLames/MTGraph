#!/usr/bin/env python
"""WCC + SSSP benchmark: 4 datasets × 3 GPU configs."""
import subprocess, re, os, sys, json

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

GPUS = [2, 4, 8]
PORT = 36000

def run_one(algo, name, path, is_long, ngpu):
    global PORT
    PORT += 1
    timeout = 900 if name in ("sk-2005", "uk-2007") else 300
    script = os.path.join(PROJ, f"src/multigpu/scripts/{algo}.py")
    cmd = [
        TORCHRUN, f"--nproc_per_node={ngpu}",
        "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
        script, "--data_path", path, "-n", str(ngpu), "-l", is_long,
        "-i", "50",  # max iters (algorithm will converge earlier)
    ]
    if algo == "sssp":
        cmd += ["-s", "0"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ)
        out = r.stdout + r.stderr
        result = {}
        # 提取 one-iter time
        for line in out.splitlines():
            m = re.search(r"one-iter time ([\d.eE+-]+)", line)
            if m and "Rank 0" in line:
                result["per_iter"] = float(m.group(1))
        # 提取算法特定指标
        for line in out.splitlines():
            if "Rank 0" in line:
                if algo == "wcc":
                    m = re.search(r"components (\d+)", line)
                    if m: result["components"] = int(m.group(1))
                    m = re.search(r"iters (\d+)", line)
                    if m: result["iters"] = int(m.group(1))
                elif algo == "sssp":
                    m = re.search(r"maxdist ([\d.]+)", line)
                    if m: result["maxdist"] = float(m.group(1))
                    m = re.search(r"reachable (\d+)", line)
                    if m: result["reachable"] = int(m.group(1))
                    m = re.search(r"iters (\d+)", line)
                    if m: result["iters"] = int(m.group(1))
        if result.get("per_iter"):
            return result
        return None
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

# 之前 PR 和 BFS 的结果（复用，不重跑）
PR_RESULTS = {
    "soc-twitter":   {2: 0.0685, 4: 0.0767, 8: 0.1336},
    "soc-sinaweibo": {2: 0.1041, 4: 0.0868, 8: 0.1279},
    "sk-2005":       {2: 0.3011, 4: 0.2661, 8: 0.3067},
    "uk-2007":       {2: None,   4: 0.1644, 8: 0.1914},
}

BFS_RESULTS = {
    "soc-twitter":   {2: {"maxVal": 7, "nonVisited": 462416, "time": 1.1},
                      4: {"maxVal": 9, "nonVisited": 907219, "time": 2.2},
                      8: {"maxVal": 8, "nonVisited": 915954, "time": 3.4}},
    "soc-sinaweibo": {2: {"maxVal": 3, "nonVisited": 1318, "time": 1.8},
                      4: {"maxVal": 5, "nonVisited": 32780845, "time": 2.2},
                      8: {"maxVal": 11, "nonVisited": 46490600, "time": 3.4}},
    "sk-2005":       {2: {"maxVal": 6, "nonVisited": 65315, "time": 6.8},
                      4: {"maxVal": 8, "nonVisited": 52863, "time": 10.8},
                      8: {"maxVal": 9, "nonVisited": 20611, "time": 11.9}},
    "uk-2007":       {2: None,
                      4: {"maxVal": 6, "nonVisited": 958137, "time": 16.4},
                      8: {"maxVal": 7, "nonVisited": 933537, "time": 19.1}},
}

all_results = {"PR": PR_RESULTS, "BFS": BFS_RESULTS, "WCC": {}, "SSSP": {}}

# 跑 WCC 和 SSSP
for algo in ["wcc", "sssp"]:
    for name, path, is_long in DATASETS:
        all_results[algo.upper()][name] = {}
        for ngpu in GPUS:
            sys.stdout.write(f"  {algo.upper()} {name} {ngpu}-GPU ... ")
            sys.stdout.flush()
            r = run_one(algo, name, path, is_long, ngpu)
            all_results[algo.upper()][name][ngpu] = r
            if r is None or isinstance(r, str):
                print(r if r else "FAIL")
            else:
                print(f"per_iter={r.get('per_iter', '?'):.4f}s {r}")

# 输出完整汇总表格
print("\n" + "=" * 90)
print("Complete Benchmark Summary (4 algorithms × 4 datasets × 3 GPU configs)")
print("=" * 90)

for algo in ["PR", "BFS", "WCC", "SSSP"]:
    print(f"\n--- {algo} ---")
    print(f"{'Dataset':<16} {'2-GPU':>16} {'4-GPU':>16} {'8-GPU':>16}")
    print("-" * 64)
    for name, _, _ in DATASETS:
        row = f"{name:<16}"
        for ngpu in GPUS:
            r = all_results[algo].get(name, {}).get(ngpu)
            if r is None:
                row += f" {'FAIL':>16}"
            elif isinstance(r, str):
                row += f" {r:>16}"
            elif algo == "PR":
                row += f" {r*1000:>10.1f} ms"
            elif algo == "BFS":
                row += f" mv={r['maxVal']} t={r['time']:.1f}s"
            elif algo == "WCC":
                row += f" {r.get('per_iter',0)*1000:.0f}ms comp={r.get('components','?')}"
            elif algo == "SSSP":
                row += f" {r.get('per_iter',0)*1000:.0f}ms md={r.get('maxdist','?')}"
        print(row)

# 保存 JSON 结果
with open(os.path.join(PROJ, "benchmark_results.json"), "w") as f:
    def default(o):
        if isinstance(o, dict): return o
        return str(o)
    json.dump(all_results, f, indent=2, default=default)
print(f"\nDetailed results saved to benchmark_results.json")
