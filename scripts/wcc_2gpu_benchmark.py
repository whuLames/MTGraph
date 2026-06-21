#!/usr/bin/env python
"""WCC 2-GPU benchmark on 4 datasets."""
import subprocess, re, os, sys

TORCHRUN = "/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun"
PROJ = "/home/zyl/huawei/torchMTGraph"
PORT = 38000

DATASETS = [
    ("soc-twitter",   "/home/zyl/data/csr_data/soc-twitter",   "0"),
    ("soc-sinaweibo", "/home/zyl/data/csr_data/soc-sinaweibo", "0"),
    ("sk-2005",       "/home/zyl/data/csr_data/sk-2005",       "1"),
    ("uk-2007",       "/home/zyl/data/csr_data/uk-2007",       "1"),
]

results = {}
for name, path, is_long in DATASETS:
    PORT += 1
    sys.stdout.write(f"  {name} 2-GPU ... ")
    sys.stdout.flush()
    cmd = [
        TORCHRUN, "--nproc_per_node=2",
        "--rdzv_backend=c10d", f"--rdzv_endpoint=localhost:{PORT}",
        os.path.join(PROJ, "src/multigpu/scripts/wcc.py"),
        "--data_path", path, "-n", "2", "-l", is_long, "-i", "50",
    ]
    timeout = 900 if name in ("sk-2005", "uk-2007") else 300
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJ)
        out = r.stdout + r.stderr
        result = {}
        for line in out.splitlines():
            if "Rank 0" in line:
                m = re.search(r"one-iter time ([\d.eE+-]+)", line)
                if m: result["per_iter"] = float(m.group(1))
                m = re.search(r"components (\d+)", line)
                if m: result["components"] = int(m.group(1))
                m = re.search(r"iters (\d+)", line)
                if m: result["iters"] = int(m.group(1))
        if result.get("per_iter"):
            results[name] = result
            print(f"{result['per_iter']*1000:.1f} ms/iter, comp={result['components']}, iters={result['iters']}")
        else:
            results[name] = "FAIL"
            print("FAIL")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"ERROR: {e}")

# 打印表格 + 写入文件
lines = []
lines.append("# WCC 2-GPU Benchmark Results")
lines.append("")
lines.append("| Dataset | Vertices | Edges | ms/iter | Components | Converge iters |")
lines.append("|---|---:|---:|---:|---:|---:|")
sizes = {"soc-twitter": "21.3M / 530M", "soc-sinaweibo": "58.7M / 523M",
         "sk-2005": "50.6M / 3.62B", "uk-2007": "105.2M / 7.46B"}
for name, path, is_long in DATASETS:
    r = results.get(name)
    if isinstance(r, dict):
        lines.append(f"| {name} | {sizes[name]} | | {r['per_iter']*1000:.1f} | {r['components']} | {r['iters']} |")
    else:
        lines.append(f"| {name} | {sizes[name]} | | FAIL | - | - |")

output = "\n".join(lines) + "\n"
print("\n" + output)
with open(os.path.join(PROJ, "WCC_2GPU_RESULTS.md"), "w") as f:
    f.write(output)
print(f"Saved to WCC_2GPU_RESULTS.md")
