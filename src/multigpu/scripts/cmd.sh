#!/bin/bash
# torch.distributed 环境变量初始化（用于脚本式多 GPU）。
# 用法：source cmd.sh [world_size] [rank]
# 若未指定，MASTER_ADDR/MASTER_PORT 从环境读取，缺失时给默认值。

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12345}"

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
