#!/bin/bash
# PageRank benchmark: soc-sinaweibo / soc-twitter / sk-2005 / uk-2007
# 10 轮迭代，取每轮平均时间

export PATH=/usr/bin:/usr/local/bin:/home/zyl/.conda/envs/torch_mtgraph/bin:$PATH
set -u
cd "$(dirname "$0")/.."

TORCHRUN=/home/zyl/.conda/envs/torch_mtgraph/bin/torchrun
RESULTS_DIR=/tmp/pr_benchmark
mkdir -p $RESULTS_DIR
PORT=31000

# 数据集配置: name|path|is_long
DATASETS=(
    "soc-sinaweibo|/home/zyl/data/csr_data/soc-sinaweibo|0"
    "soc-twitter|/home/zyl/data/csr_data/soc-twitter|0"
    "sk-2005|/home/zyl/data/csr_data/sk-2005|1"
    "uk-2007|/home/zyl/data/csr_data/uk-2007|1"
)

run_pr() {
    local NAME=$1 PATH=$2 IS_LONG=$3 NGPU=$4
    PORT=$((PORT+1))
    local OUT=$RESULTS_DIR/${NAME}_${NGPU}gpu.log
    local TIMEOUT=600

    # uk-2007 读图大，加长 timeout
    if [ "$NAME" = "uk-2007" ]; then TIMEOUT=900; fi
    if [ "$NAME" = "sk-2005" ] && [ "$NGPU" -ge 8 ]; then TIMEOUT=900; fi

    timeout $TIMEOUT $TORCHRUN --nproc_per_node=$NGPU \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
        src/multigpu/scripts/pr.py \
        --data_path "$PATH" -n $NGPU -i 10 -l $IS_LONG 2>&1 | \
        grep -E "one-iter time" > $OUT 2>&1

    # 提取 one-iter time（取所有 rank 的最大值作为 wall-clock 估计）
    if [ -s "$OUT" ]; then
        # 格式: "Rank X has data Y after 10 all reduce one-iter time Z"
        # 取 rank 0 的 one-iter time
        local time=$(grep "Rank 0" $OUT | grep -oE "one-iter time [0-9.]+" | grep -oE "[0-9.]+$")
        echo "$time"
    else
        echo "FAIL"
    fi
}

echo "dataset,2-GPU,4-GPU,8-GPU" > $RESULTS_DIR/results.csv

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r NAME PATH IS_LONG <<< "$entry"
    echo "======== $NAME (is_long=$IS_LONG) ========" | tee -a $RESULTS_DIR/detail.log

    TIMES=""
    for NGPU in 2 4 8; do
        echo " Running $NGPU-GPU..." | tee -a $RESULTS_DIR/detail.log
        T=$(run_pr "$NAME" "$PATH" "$IS_LONG" "$NGPU")
        echo "  $NGPU-GPU: one-iter time = ${T}s" | tee -a $RESULTS_DIR/detail.log
        TIMES="${TIMES}${T},"
    done
    TIMES="${TIMES%,}"
    echo "$NAME,$TIMES" >> $RESULTS_DIR/results.csv
    echo "" | tee -a $RESULTS_DIR/detail.log
done

echo ""
echo "==================== FINAL RESULTS ===================="
echo ""
cat $RESULTS_DIR/results.csv
