#!/usr/bin/env python
"""PageRank benchmark: 4 datasets × 3 GPU configs, 10 iters each."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"

DATASETS = [
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

GPUS = [2, 4, 8]
PORT = 32000

def run_one(name, path, is_long, ngpu):
    global PORT
    PORT += 1
    timeout = 900 if name in ("sk-2005", "uk-2007") else 300

    cmd = [
        TORCHRUN, f"--nproc_per_node={ngpu}",
        "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
        os.path.join(PROJ, "src/multigpu/scripts/pr.py"),
        "--data_path", path, "-n", str(ngpu), "-i", "10", "-l", is_long,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ)
        out = r.stdout + r.stderr
        # 提取 rank 0 的 one-iter time
        for line in out.splitlines():
            if "Rank 0" in line and "one-iter time" in line:
                m = re.search(r"one-iter time ([\d.]+)", line)
                if m:
                    return float(m.group(1))
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
        t = run_one(name, path, is_long, ngpu)
        results[name][ngpu] = t
        print(f"{t}")

# 打印表格
print("\n" + "=" * 60)
print("PageRank Benchmark Results (avg ms/iter over 10 iters)")
print("=" * 60)
print(f"{'Dataset':<20} {'2-GPU':>12} {'4-GPU':>12} {'8-GPU':>12}")
print("-" * 60)
for name, _, _ in DATASETS:
    row = name
    for ngpu in GPUS:
        t = results[name][ngpu]
        if t is None or isinstance(t, str):
            val = t if t else "FAIL"
        else:
            val = f"{t*1000:.1f} ms"
        row += f" {val:>12}"
    print(row)
print("=" * 60)
