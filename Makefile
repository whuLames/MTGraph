# torchMTGraph 顶层 Makefile
PROJ_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: help install test smoke-test clean check

help:
	@echo "torchMTGraph targets:"
	@echo "  make install      - pip install -e .（在 torch_mtgraph 环境内）"
	@echo "  make test         - pytest 单元测试"
	@echo "  make smoke-test   - 多 GPU 脚本版 smoke test（torchrun）"
	@echo "  make check        - 检查硬编码清零 + import 健康度"
	@echo "  make clean        - 清理 __pycache__/build 等"

install:
	pip install -e .

test:
	python -m pytest tests/ -v

smoke-test:
	torchrun --nproc_per_node=2 $(PROJ_ROOT)/src/multigpu/scripts/pr.py \
		--data_path $(PROJ_ROOT)/tests/fixtures/small --partition_num 2 --iterations 5

check:
	@echo "==> sys.path.append 检查（应为空）"
	@grep -rnE "^[[:space:]]*sys\.path\.append\(" --include='*.py' $(PROJ_ROOT) \
		--exclude-dir=.git --exclude-dir=__pycache__ || echo "✓ 清零"
	@echo ""
	@echo "==> import 健康度"
	@python -c "from src.framework.GASProgram import GASProgram; print('✓ GASProgram')"
	@python -c "from src.framework.strategy.MultiGPUStrategyByNCCL import MultiGPUStrategyByNCCL; print('✓ MultiGPUStrategyByNCCL')"
	@python -c "from src.algorithms.BFS import BFS; print('✓ BFS')"
	@python -c "import torch_scatter; print('✓ torch_scatter')"

clean:
	find $(PROJ_ROOT) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find $(PROJ_ROOT) -name "*.pyc" -delete 2>/dev/null || true
	find $(PROJ_ROOT) -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
