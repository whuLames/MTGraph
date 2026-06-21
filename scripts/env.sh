#!/bin/bash
# MTGraphUnified 运行环境变量初始化（在运行 Python 代码前 source 之）。
#
# 主要解决：
#   1. extension.so 编译后运行时找不到 libc10.so 的问题
#   2. conda 环境 CUDA toolkit 与 PyTorch 默认 /usr/local/cuda 不一致问题
# 用法： source scripts/env.sh

# 1. torch 的核心库（libc10.so / libtorch.so 等）
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null)
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
    echo "✓ LD_LIBRARY_PATH += $TORCH_LIB_DIR"
else
    echo "⚠ 未找到 torch 安装，跳过 LD_LIBRARY_PATH 设置"
fi

# 2. CUDA_HOME：优先 conda 环境（nvcc 所在 prefix），其次系统 /usr/local/cuda
if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_DIR=$(dirname $(dirname $(which nvcc)))
        # conda 装 CUDA 时 cuda_runtime.h 在 <prefix>/targets/<arch>/include
        if [ -d "$NVCC_DIR/targets" ]; then
            export CUDA_HOME="$NVCC_DIR"
            export CUDA_PATH="$NVCC_DIR"
            echo "✓ CUDA_HOME=$CUDA_HOME（conda CUDA toolkit）"
        elif [ -d "$NVCC_DIR/include/cuda_runtime.h" ]; then
            export CUDA_HOME="$NVCC_DIR"
            export CUDA_PATH="$NVCC_DIR"
            echo "✓ CUDA_HOME=$CUDA_HOME"
        fi
    fi
    if [ -z "$CUDA_HOME" ] && [ -d /usr/local/cuda ]; then
        export CUDA_HOME="/usr/local/cuda"
        export CUDA_PATH="/usr/local/cuda"
        echo "✓ CUDA_HOME=$CUDA_HOME（系统默认）"
    fi
fi

# 3. 图数据根目录（被 .sh 脚本里的 \${GRAPH_DATA_ROOT:-...} 引用）
export GRAPH_DATA_ROOT="${GRAPH_DATA_ROOT:-/data/graphs}"
echo "✓ GRAPH_DATA_ROOT=$GRAPH_DATA_ROOT（按需修改）"

# 4. NCCL 根目录（被各 Makefile 的 \${NCCL_ROOT:-...} 引用）

echo "✓ NCCL_ROOT=$NCCL_ROOT（按需修改）"

