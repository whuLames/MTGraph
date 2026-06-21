"""
common.python 模块单元测试。

覆盖：
  - multi_arange：与朴素实现对比
  - read_data：读 fixtures/small 验证字段
  - EBP：分区数守恒 + 边数守恒 + 边均衡性
  - CommPlan：dataclass 字段默认值
"""
import os
import pytest
import torch

from src.common.python.arange import multi_arange
from src.common.python.io import read_data, readGraphBinary
from src.common.python.partition import EBP, get_partition
from src.common.python.comm_plan import CommPlan


# ---------------- multi_arange ----------------

def _multi_arange_naive(starts, ends):
    """朴素实现，用于对比。"""
    out = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        out.extend(range(s, e))
    return torch.tensor(out, dtype=starts.dtype, device=starts.device)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_multi_arange_basic(device):
    starts = torch.tensor([0, 5, 10], dtype=torch.int64, device=device)
    ends = torch.tensor([3, 8, 12], dtype=torch.int64, device=device)
    out = multi_arange(starts, ends)
    expected = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11], dtype=torch.int64, device=device)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_multi_arange_random(device):
    torch.manual_seed(0)
    starts = torch.randint(0, 100, (50,), device=device)
    sizes = torch.randint(1, 20, (50,), device=device)
    ends = starts + sizes
    assert torch.equal(multi_arange(starts, ends), _multi_arange_naive(starts, ends))


def test_multi_arange_empty():
    starts = torch.tensor([], dtype=torch.int64)
    ends = torch.tensor([], dtype=torch.int64)
    out = multi_arange(starts, ends)
    assert len(out) == 0


# ---------------- read_data ----------------

def test_read_data_small(small_graph_path):
    row_ptr, columns = read_data(small_graph_path)
    # fixtures/small 是 100 顶点 500 边
    assert len(row_ptr) == 101     # n_verts + 1
    assert len(columns) == 500     # n_edges
    # row_ptr 单调不减
    assert torch.all(row_ptr[1:] >= row_ptr[:-1])
    # columns 在 [0, n_verts) 范围内
    assert columns.min() >= 0
    assert columns.max() < 100


def test_readGraphBinary_alias(small_graph_path):
    """readGraphBinary 应等价于 read_data 默认参数。"""
    a = read_data(small_graph_path)
    b = readGraphBinary(small_graph_path)
    assert len(a) == len(b) == 2
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_read_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_data(str(tmp_path / "nonexistent"))


# ---------------- EBP ----------------

@pytest.mark.parametrize("nparts", [1, 2, 4, 8])
def test_ebp_edge_conservation(small_graph_path, nparts):
    """所有分区的边数之和应等于总边数。"""
    row_ptr, columns = read_data(small_graph_path)
    parts_row_ptr, parts_columns, vbidx = EBP(row_ptr, columns, nparts)
    assert len(parts_row_ptr) == nparts
    assert len(parts_columns) == nparts
    assert len(vbidx) == nparts + 1
    total_part_edges = sum(len(c) for c in parts_columns)
    assert total_part_edges == len(columns)


@pytest.mark.parametrize("nparts", [2, 4, 8])
def test_ebp_balance(small_graph_path, nparts):
    """各分区边数应大致均衡（最大分区 / 最小分区 ≤ 2.0）。"""
    row_ptr, columns = read_data(small_graph_path)
    _, parts_columns, _ = EBP(row_ptr, columns, nparts)
    sizes = [len(c) for c in parts_columns]
    # 当 nparts > n_verts 时会出现 0，跳过那种情况
    if min(sizes) > 0:
        ratio = max(sizes) / min(sizes)
        assert ratio <= 2.0, f"imbalance: sizes={sizes}, ratio={ratio}"


def test_ebp_partition_alias():
    """get_partition 应是 EBP 的别名。"""
    assert get_partition is EBP


def test_ebp_nparts_1(small_graph_path):
    """nparts=1 时返回整体图。"""
    row_ptr, columns = read_data(small_graph_path)
    pr, pc, vbidx = EBP(row_ptr, columns, 1)
    assert len(pc[0]) == len(columns)


# ---------------- CommPlan ----------------

def test_comm_plan_defaults():
    cp = CommPlan()
    assert cp.send_counts == []
    assert cp.send_offsets == []
    assert cp.recv_offsets == []
    assert cp.recv_totals == []


def test_comm_plan_fields():
    cp = CommPlan(
        send_counts=[[1, 2], [3, 4]],
        send_offsets=[[0, 1], [0, 3]],
        recv_offsets=[[0, 2], [1, 4]],
        recv_totals=[5, 6],
    )
    assert cp.send_counts[0][1] == 2
    assert cp.recv_totals[1] == 6
