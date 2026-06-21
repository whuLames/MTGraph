"""
multigpu/scripts 静态检查（不实际运行多 GPU）。

覆盖：
  - bfs.py / pr.py / run.py / test.py 的 ast 解析
  - 关键函数存在性（bfs_bc / pr / get_partition / read_data / multi_arange import 自 common）
  - 不再有内部 multi_arange / read_data / get_partition / allreduce 定义
  - 不含硬编码 IP / 路径
"""
import ast
import os
import pytest

SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "multigpu", "scripts"
)


def _parse(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _names(tree):
    return {n.name for n in ast.iter_child_nodes(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _imports_from(tree, module_prefix):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(module_prefix):
                return True
    return False


@pytest.mark.parametrize("filename", ["bfs.py", "pr.py", "run.py", "test.py"])
def test_script_parses(filename):
    _parse(filename)


def test_bfs_has_bfs_bc():
    tree = _parse("bfs.py")
    names = _names(tree)
    assert "bfs_bc" in names, "bfs.py 应保留 bfs_bc（含本地多轮迭代的 direction-optimizing 变体）"
    assert "bfs" in names, "bfs.py 应保留 bfs（单层变体）"


def test_bfs_no_internal_duplicates():
    tree = _parse("bfs.py")
    names = _names(tree)
    for dup in ["multi_arange", "read_data", "get_partition", "allreduce"]:
        assert dup not in names, f"{dup} 不应在 bfs.py 内部定义（已抽到 common）"


def test_bfs_imports_from_common():
    tree = _parse("bfs.py")
    assert _imports_from(tree, "src.common.python")


def test_pr_has_pr_function():
    tree = _parse("pr.py")
    assert "pr" in _names(tree)


def test_pr_no_internal_duplicates():
    tree = _parse("pr.py")
    names = _names(tree)
    for dup in ["multi_arange", "read_data", "get_partition", "allreduce"]:
        assert dup not in names


def test_pr_imports_from_common():
    tree = _parse("pr.py")
    assert _imports_from(tree, "src.common.python")


@pytest.mark.parametrize("filename", ["bfs.py", "pr.py", "run.py", "test.py"])
def test_no_hardcoded_master_addr(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    with open(path) as f:
        content = f.read()
    for forbidden in ["10.230.52.203", "111.6.235.231"]:
        assert forbidden not in content, f"{filename} 残留硬编码 IP: {forbidden}"
