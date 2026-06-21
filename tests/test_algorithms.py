"""
5 个算法的静态检查 + 算法级功能验证。

覆盖：
  - BFS / PageRank / ShortestPaths(SSSP) / ConnectedComponents(WCC) / HITS
  - 类存在性 + 继承自 GASProgram
  - 关键方法（gather/sum/apply/scatter）存在
  - 在 fixtures/small 上的最小功能验证
"""
import os
import ast
import pytest
import torch

from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph


ALGORITHMS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "algorithms")


# ---------------- 静态检查 ----------------

@pytest.mark.parametrize("filename,expected_class", [
    ("BFS.py", "BFS"),
    ("PageRank.py", "PageRank"),
    ("ShortestPaths.py", "SSSP"),
    ("ConnectedComponents.py", "ConnectedComponents"),
    ("HITS.py", "HITS"),
])
def test_algorithm_class_exists(filename, expected_class):
    """5 个算法文件应包含预期的类定义。"""
    path = os.path.join(ALGORITHMS_DIR, filename)
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    classes = [n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)]
    assert expected_class in classes, f"{filename} should define class {expected_class}, got {classes}"


def test_algorithms_inherit_gasprogram():
    """5 个算法都应继承 GASProgram。"""
    from src.algorithms.BFS import BFS
    from src.algorithms.PageRank import PageRank
    from src.algorithms.ShortestPaths import SSSP
    from src.algorithms.ConnectedComponents import ConnectedComponents
    from src.algorithms.HITS import HITS

    for cls in (BFS, PageRank, SSSP, ConnectedComponents, HITS):
        assert issubclass(cls, GASProgram), f"{cls.__name__} should inherit from GASProgram"


@pytest.mark.parametrize("filename,expected_methods", [
    ("BFS.py", ["gather", "sum", "apply", "scatter"]),
    ("PageRank.py", ["gather", "sum", "apply", "scatter"]),
    ("ShortestPaths.py", ["gather", "sum", "apply", "scatter"]),
    ("ConnectedComponents.py", ["gather", "sum", "apply", "scatter"]),
    ("HITS.py", ["gather", "sum", "apply", "scatter"]),
])
def test_algorithm_has_gas_methods(filename, expected_methods):
    """每个算法应实现 GAS 的 4 个核心方法。"""
    path = os.path.join(ALGORITHMS_DIR, filename)
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    methods = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.add(item.name)
    for m in expected_methods:
        assert m in methods, f"{filename} missing method: {m}"


# ---------------- 功能验证（在 fixtures/small 上）----------------

@pytest.fixture(scope="module")
def small_graph():
    """加载 fixtures/small 作为 CSRCGraph。"""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "small")
    return CSRCGraph.read_csrc_graph_bin(fixtures_dir)


def test_pagerank_initialization(small_graph):
    """PageRank 应能正确初始化。"""
    from src.algorithms.PageRank import PageRank
    pr = PageRank(small_graph, num_iter=5)
    assert pr.vertex_data.shape[0] == small_graph.num_vertices
    assert pr.vertex_data.shape[1] == 2  # rank + delta


def test_bfs_initialization(small_graph):
    """BFS 应能正确初始化（source 顶点距离设为 0）。"""
    from src.algorithms.BFS import BFS
    bfs = BFS(small_graph, start_from=[0])
    assert bfs.vertex_data.shape[0] == small_graph.num_vertices
    assert bfs.vertex_data[0].item() == 0  # source 距离为 0


def test_sssp_initialization(small_graph):
    """SSSP 应能正确初始化（需要 source + edge_data）。"""
    from src.algorithms.ShortestPaths import SSSP
    edge_data = torch.ones(small_graph.num_edges, dtype=torch.float32)
    sssp = SSSP(small_graph, source=0, edge_data=edge_data)
    assert sssp.vertex_data.shape[0] == small_graph.num_vertices
    assert sssp.vertex_data[0].item() == 0.0  # source 距离为 0


def test_cc_initialization(small_graph):
    """WCC（ConnectedComponents）应能正确初始化。"""
    from src.algorithms.ConnectedComponents import ConnectedComponents
    cc = ConnectedComponents(small_graph)
    assert cc.vertex_data.shape[0] == small_graph.num_vertices


def test_hits_initialization(small_graph):
    """HITS 应能正确初始化。"""
    from src.algorithms.HITS import HITS
    hits = HITS(max_steps=5, graph=small_graph, num_iter=5)
    assert hits.vertex_data.shape[0] == small_graph.num_vertices
