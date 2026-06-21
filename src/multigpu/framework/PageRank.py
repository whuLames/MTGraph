"""
多 GPU 框架版 PageRank（基于 GASProgram + MultiGPUStrategyByNCCL）。

与 src/algorithms/PageRank.py 的差异：
  - 算法（gather/sum/apply/scatter）一致；
  - 这里使用 MultiGPUStrategyByNCCL（torch.distributed NCCL）执行多 GPU 计算；
  - 通过 torch.multiprocessing spawn 多进程，每进程绑定一个 GPU；
  - 启动方式：python src/multigpu/framework/PageRank.py --graph <path> --device_num <N>
"""

import sys
import torch
import argparse
import time
import logging
from torch_scatter import segment_csr
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.framework.helper import batched_csr_selection
from src.framework.strategy.MultiGPUStrategyByNCCL import MultiGPUStrategyByNCCL
from src.framework.partition.GeminiPartition import GeminiPartition

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class PageRank(GASProgram):
    def __init__(self, graph: CSRCGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, num_iter=20,
                 **kwargs):
        # vertex_data: [num_vertices, 2]. rank and delta
        vertex_data = None if graph == None else torch.ones((graph.num_vertices, 2), dtype=vertex_data_type)
        self.UPDATE_THRESHOLD = 0.001
        self.ITER_THRESHOLD = num_iter
        self.out_degree = None
        super().__init__(graph, vertex_data=vertex_data, vertex_data_type=vertex_data_type,
                         edge_data_type=edge_data_type, **kwargs)

    def gather(self, vertices, nbrs, edges, ptr):
        if self.nbr_update_freq == 0:
            if self.out_degree is None:
                self.out_degree = 1 / self.graph.out_degree(nbrs).to(self.device)
        else:
            self.out_degree = 1 / self.graph.out_degree(nbrs).to(self.device)
        data = self.vertex_data[nbrs, 0] * self.out_degree
        return data, ptr

    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='sum')

    def apply(self, vertices, gathered_sum):
        rnew = 0.15 + 0.85 * gathered_sum
        delta = torch.abs(rnew - self.vertex_data[vertices, 0])
        return torch.stack([rnew, delta], dim=-1), torch.ones(vertices.shape + (2,), dtype=torch.bool,
                                                              device=self.device)

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if self.nbr_update_freq > 0:
            delta = self.vertex_data[vertices, 1]
            selected = torch.where(delta > self.UPDATE_THRESHOLD)[0]
            if selected.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD - 1:
                starts, ends = ptr[selected], ptr[selected + 1]
                result, ptr = batched_csr_selection(starts, ends)
                all_neighbors = nbrs[result]
                self.activate(all_neighbors)
        else:
            if self.curr_iter < self.ITER_THRESHOLD - 1:
                self.not_change_activated_next_iter()
        return None, None

    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            in_nbrs, ptr = self.graph.all_in_nbrs_csr()
        else:
            in_nbrs, ptr = self.graph.in_nbrs_csr(vertices)
        return in_nbrs, None, ptr

    def scatter_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph dir (csr_vlist[.bin] / csr_elist[.bin])',
                        required=True)
    parser.add_argument('--device_num', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--num_iter', type=int, default=20, help='PageRank iterations')
    args = parser.parse_args()

    print('reading graph...', end=' ', flush=True)
    graph = CSRCGraph.read_csrc_graph_bin(args.graph)
    graph.pin_memory()
    print(f'V={graph.num_vertices}, E={graph.num_edges}. Done!')

    pr = PageRank(graph, num_iter=args.num_iter)
    partition = GeminiPartition(graph, num_partitions=args.device_num,
                                alpha=8 * args.device_num - 8)
    strategy = MultiGPUStrategyByNCCL(pr, partition, device_num=args.device_num)

    t1 = time.time()
    v_data, _ = strategy.compute()
    t2 = time.time()
    print(f'multi-GPU compute time: {t2 - t1:.3f}s')


if __name__ == '__main__':
    main()
