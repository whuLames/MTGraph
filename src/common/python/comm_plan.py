"""
CommPlan：通信计划数据结构（与 CUDA 端 struct CommPlan 字段对齐）。

来源：MTGraph/src/bfs/bfs_multigpu.cu:109 的 struct CommPlan 权威定义。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CommPlan:
    """通信计划。

    Attributes:
        send_counts:  List[List[int]]    [src][dst]    要发送的元素个数
        send_offsets: List[List[int]]    [src][dst]    源缓冲区偏移
        recv_offsets: List[List[int]]    [dst][src]    目标缓冲区偏移
        recv_totals:  List[int]          每个 rank 接收的总元素数
    """
    send_counts: List[List[int]] = field(default_factory=list)
    send_offsets: List[List[int]] = field(default_factory=list)
    recv_offsets: List[List[int]] = field(default_factory=list)
    recv_totals: List[int] = field(default_factory=list)
