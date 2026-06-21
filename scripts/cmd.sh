#!/bin/bash
# torchMTGraph 多 GPU 启动辅助脚本（torchrun）。
#
# 用法：
#   source scripts/cmd.sh                  # 默认 2 GPU + fixtures/small
#   source scripts/cmd.sh 4 path/to/graph  # 自定义 GPU 数与图路径
#
# 也可单独使用 torchrun 直接启动：
#   torchrun --nproc_per_node=2 src/multigpu/scripts/pr.py \
#       --data_path tests/fixtures/small --partition_num 2 --iterations 20

NGPU="${1:-2}"
GRAPH="${2:-$(dirname $(dirname $0))/tests/fixtures/small}"
ITERS="${3:-20}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "启动 PageRank 多 GPU: ngpu=$NGPU graph=$GRAPH iters=$ITERS"
torchrun --nproc_per_node=$NGPU \
    "$(dirname $(dirname $0))/src/multigpu/scripts/pr.py" \
    --data_path "$GRAPH" --partition_num $NGPU --iterations $ITERS
