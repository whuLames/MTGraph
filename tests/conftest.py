"""
pytest 共享 fixture。
"""
import os
import pytest


@pytest.fixture(scope="session")
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="session")
def small_graph_path(fixtures_dir):
    """tests/fixtures/small/，含 csr_vlist.bin + csr_elist.bin（100 顶点 500 边）。"""
    return os.path.join(fixtures_dir, "small")
