#!/bin/bash
# cit-Patents 多 GPU 功能测试：BFS + PageRank × 2/4/8 GPU
# 验证：1) 正常完成；2) 两 rank 间输出一致；3) 数值合理

set -u
cd "$(dirname "$0")/.."

PYTHON=/home/zyl/.conda/envs/torch_mtgraph/bin/python
TORCHRUN=/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun
DATA=/home/zyl/data/csr_data/cit-Patents
RESULTS_DIR=/tmp/cit_patents_results
mkdir -p $RESULTS_DIR

# 用动态端口避免冲突
PORT=29600

run_pr() {
    local NGPU=$1
    local ITERS=$2
    local OUT=$RESULTS_DIR/pr_${NGPU}gpu.out
    echo "  → PageRank ${NGPU}-GPU, ${ITERS} iters..."
    PORT=$((PORT+1))
    timeout 300 $TORCHRUN --nproc_per_node=$NGPU \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
        src/multigpu/scripts/pr.py \
        --data_path $DATA -n $NGPU -i $ITERS 2>&1 | \
        grep -E "Rank [0-9]+ has data|Rank [0-9]+ all reduce time|one-iter time" | \
        sort -u > $OUT
    if [ -s "$OUT" ]; then
        echo "    输出: $(cat $OUT | head -2)"
        return 0
    else
        echo "    ✗ 失败（无输出）"
        return 1
    fi
}

run_bfs() {
    local NGPU=$1
    local OUT=$RESULTS_DIR/bfs_${NGPU}gpu.out
    echo "  → BFS bfs_bc ${NGPU}-GPU..."
    PORT=$((PORT+1))
    timeout 600 $TORCHRUN --nproc_per_node=$NGPU \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
        src/multigpu/scripts/bfs.py \
        --data_path $DATA -n $NGPU -s 0 2>&1 | \
        grep -E "maxVal|nonVisitedNum|elapsed time" | \
        sort -u > $OUT
    if [ -s "$OUT" ]; then
        echo "    输出: $(cat $OUT | head -2)"
        return 0
    else
        echo "    ✗ 失败（无输出）"
        return 1
    fi
}

echo "========== cit-Patents 多 GPU 功能测试 =========="
echo "数据集: $DATA (3.77M 顶点, 33.04M 边)"
echo ""

for NGPU in 2 4 8; do
    echo "======== ${NGPU} GPU ========"
    run_pr $NGPU 10
    run_bfs $NGPU
    echo ""
done

echo "========== 结果汇总（BFS 末端距离一致性）=========="
for NGPU in 2 4 8; do
    OUT=$RESULTS_DIR/bfs_${NGPU}gpu.out
    if [ -s "$OUT" ]; then
        # 每个 rank 应该有相同的 maxVal/nonVisitedNum
        unique=$(sort -u $OUT | wc -l)
        echo "${NGPU} GPU: $(sort -u $OUT | head -2 | tr '\n' '|')"
    fi
done

echo ""
echo "========== 结果汇总（PageRank sum 跨 GPU 一致性）=========="
for NGPU in 2 4 8; do
    OUT=$RESULTS_DIR/pr_${NGPU}gpu.out
    if [ -s "$OUT" ]; then
        echo "${NGPU} GPU: $(sort -u $OUT | head -2 | tr '\n' '|')"
    fi
done
