#!/usr/bin/env python
"""BFS single-layer benchmark (push/pull, no inner loop): 4 datasets × 3 GPU."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"
PORT = 40000

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]
GPUS = [2, 4, 8]

results = {}
for name, path, is_long in DATASETS:
    results[name] = {}
    for ngpu in GPUS:
        PORT += 1
        sys.stdout.write(f"  {name} {ngpu}-GPU ... ")
        sys.stdout.flush()
        cmd = [
            TORCHRUN, f"--nproc_per_node={ngpu}",
            "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
            os.path.join(PROJ, "src/multigpu/scripts/bfs.py"),
            "--data_path", path, "-n", str(ngpu), "-s", "0", "-l", is_long,
        ]
        env = dict(os.environ, BFS_MODE="single")
        timeout = 900 if name in ("sk-2005", "uk-2007") else 300
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ, env=env)
            out = r.stdout + r.stderr
            result = {}
            for line in out.splitlines():
                if "Rank 0" in line:
                    m = re.search(r"one-iter time ([\d.eE+-]+)", line)
                    if m: result["per_iter"] = float(m.group(1))
                    m = re.search(r"maxVal (\d+)", line)
                    if m: result["maxVal"] = int(m.group(1))
                    m = re.search(r"nonVisitedNum (\d+)", line)
                    if m: result["unvisited"] = int(m.group(1))
                    m = re.search(r"iters (\d+)", line)
                    if m: result["iters"] = int(m.group(1))
            if result.get("per_iter"):
                results[name][ngpu] = result
                print(f"{result['per_iter']*1000:.1f} ms/iter, mv={result['maxVal']}, iters={result['iters']}")
            else:
                results[name][ngpu] = "FAIL"
                print("FAIL")
        except Exception as e:
            results[name][ngpu] = f"ERROR: {e}"
            print(f"ERROR")

# 写表格
lines = ["# BFS Single-Layer (push/pull, no inner loop) Benchmark", "",
         "| Dataset | GPU | ms/iter | maxVal | unvisited | iters |",
         "|---|---:|---:|---:|---:|---:|"]
for name, _, _ in DATASETS:
    for ngpu in GPUS:
        r = results[name].get(ngpu)
        if isinstance(r, dict):
            lines.append(f"| {name} | {ngpu} | {r['per_iter']*1000:.1f} | {r['maxVal']} | {r['unvisited']} | {r['iters']} |")
        else:
            lines.append(f"| {name} | {ngpu} | FAIL | - | - | - |")

output = "\n".join(lines) + "\n"
print("\n" + output)
with open(os.path.join(PROJ, "BFS_SINGLE_RESULTS.md"), "w") as f:
    f.write(output)
print("Saved to BFS_SINGLE_RESULTS.md")
