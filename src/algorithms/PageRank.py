"""
Implementation of PageRank algorithm using the GAS framework.
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
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
from src.framework.strategy.MultiGPUStrategyByNCCL import MultiGPUStrategyByNCCL
# MultiGPUStrategyByCupyAndNCCL 已舍弃（cupy 版，不在本项目内）
# MultiGPUComputeStrategy 已舍弃（MPI 版，不在本项目内）
from src.framework.strategy.PartitionStrategy import PartitionStrategy
from src.framework.partition.VertexPartition import VertexPartition 
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
# from viztracer import VizTracer

class PageRank(GASProgram):
    def __init__(self, graph: CSRCGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, num_iter=50,
                 **kwargs):
        # vertex_data: [num_vertices, 2]. rank and delta
        vertex_data = torch.ones((graph.num_vertices, 2), dtype=vertex_data_type)
        # vertex_data /= graph.num_vertices
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
        # print('data: {} out_degree: {}'.format(self.vertex_data[nbrs, 0], self.out_degree))
        data = self.vertex_data[nbrs, 0] * self.out_degree
        return data, ptr

    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='sum')

    def apply(self, vertices, gathered_sum):
        """
        Returns:
            apply_data: 
            apply_mask: 
        """
        # logging.info('gathered_sum: {}'.format(gathered_sum))
        # logging.info('vertices: {}'.format(vertices))
        
        rnew = 0.15 + 0.85 * gathered_sum
        delta = torch.abs(rnew - self.vertex_data[vertices, 0])
        self.vertex_data[vertices, 0] = rnew
        self.vertex_data[vertices, 1] = delta
        return None, None
        # return torch.stack([rnew, delta], dim=-1), torch.ones(vertices.shape + (2,), dtype=torch.bool,
        #                                                       device=self.device)

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if self.nbr_update_freq > 0:
            delta = self.vertex_data[vertices, 1]
            selected = torch.where(delta > self.UPDATE_THRESHOLD)[0]
            if selected.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD - 1:
                # get neighbors of selected
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
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()

    print('reading graph...', end=' ', flush=True)
    # 优先使用 read_csrc_graph_bin（支持 csr_vlist.bin 目录格式 + 自动构造 CSC）
    graph = CSRCGraph.read_csrc_graph_bin(args.graph)
    print('Done!')

    # 用 SimpleStrategy（不需要 partition + csr_subgraph，CSRCGraph.shuffle_ptr 可缺省）
    pr = PageRank(graph, num_iter=50)
    strategy = SimpleStrategy(pr)
    if args.cuda:
        # 计算策略中 移至对应设备
        pr.to('cuda')
        # graph.to('cuda')
        print('use cuda')

    # tracer = VizTracer()
    t1 = time.time()
    # tracer.start()
    v_data, _ = pr.compute(strategy)
    torch.cuda.synchronize()
    # tracer.stop()
    t2 = time.time()
    # tracer.save()
    print(v_data)
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    # with open(args.output, 'w') as f:
    #     for i in range(len(v_data[:, 0])):
    #         f.write(str(v_data[i, 0].item()) + '\n')


if __name__ == '__main__':
    main()