#!/usr/bin/env python
"""BFS benchmark: 4 datasets × 3 GPU configs, 10 iters functional test."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

GPUS = [2, 4, 8]
PORT = 34000

def run_one(name, path, is_long, ngpu):
    global PORT
    PORT += 1
    timeout = 900 if name in ("sk-2005", "uk-2007") else 300

    cmd = [
        TORCHRUN, f"--nproc_per_node={ngpu}",
        "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
        os.path.join(PROJ, "src/multigpu/scripts/bfs.py"),
        "--data_path", path, "-n", str(ngpu), "-s", "0", "-l", is_long,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ)
        out = r.stdout + r.stderr
        # 提取 maxVal / nonVisitedNum（所有 rank 应一致）
        maxvals = set()
        for line in out.splitlines():
            m = re.search(r"maxVal (\d+) nonVisitedNum (\d+)", line)
            if m:
                maxvals.add((int(m.group(1)), int(m.group(2))))
        # 提取 elapsed time
        times = []
        for line in out.splitlines():
            m = re.search(r"elapsed time:\s+([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))

        if maxvals:
            mv, nv = sorted(maxvals)[0]
            t = max(times) if times else None
            consistent = len(maxvals) == 1
            return {"maxVal": mv, "nonVisited": nv, "time": t, "consistent": consistent}
        return None
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

results = {}
for name, path, is_long in DATASETS:
    results[name] = {}
    for ngpu in GPUS:
        sys.stdout.write(f"  {name} {ngpu}-GPU ... ")
        sys.stdout.flush()
        r = run_one(name, path, is_long, ngpu)
        results[name][ngpu] = r
        if r is None or isinstance(r, str):
            print(r if r else "FAIL")
        else:
            print(f"maxVal={r['maxVal']} unvisited={r['nonVisited']} time={r['time']:.1f}s consistent={'Y' if r['consistent'] else 'N'}")

# 打印表格
print("\n" + "=" * 80)
print("BFS Benchmark Results")
print("=" * 80)
print(f"{'Dataset':<16} {'GPU':>4} {'maxVal':>8} {'unvisited':>12} {'time(s)':>10} {'rank一致':>8}")
print("-" * 80)
for name, _, _ in DATASETS:
    for ngpu in GPUS:
        r = results[name][ngpu]
        if r is None or isinstance(r, str):
            val = r if r else "FAIL"
            print(f"{name:<16} {ngpu:>4}     {val:>20}")
        else:
            print(f"{name:<16} {ngpu:>4} {r['maxVal']:>8} {r['nonVisited']:>12} {r['time']:>10.1f} {'Y' if r['consistent'] else 'N':>8}")
    print()
print("=" * 80)
