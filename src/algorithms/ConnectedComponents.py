"""
Implementation of Connected Components using GAS. Assume that your graph is undirected. (we are calculating weak CCs anyway)
"""
import sys

from src.framework.GASProgram import GASProgram
from src.type.CSRGraph import CSRGraph
import torch
import argparse
import time
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from torch_scatter import segment_csr
import os
from src.framework.partition.GeminiPartition import GeminiPartition
from src.framework.partition.VertexPartition import VertexPartition
from src.framework.strategy.PartitionStrategy import PartitionStrategy

class ConnectedComponents(GASProgram):
    def __init__(self, graph: CSRGraph, vertex_data_type=torch.int32, edge_data_type=None):
        # vertex_data: [num_vertices].
        vertex_data = torch.arange(0, graph.num_vertices, dtype=vertex_data_type)
        # no edge data needed
        edge_data = None
        self.changed = False
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data)

    def gather(self, vertices, nbrs, edges, ptr):
        return self.vertex_data[nbrs], ptr
    
    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='min')
    
    def apply(self, vertices, gathered_sum):
        self.changed = self.vertex_data[vertices] > gathered_sum
        return gathered_sum, self.changed
    
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if torch.sum(self.changed) > 0:
            if self.nbr_update_freq != 0:
                changed_v = vertices[self.changed]
                changed_v_nbrs, _ = self.graph.out_nbrs_csr[changed_v]
                self.activate(changed_v_nbrs)
            else:
                self.not_change_activated_next_iter()
        return None, None
    
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0: 
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr
    
    def scatter_nbrs(self, vertices):
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph = CSRGraph.read_graph_bin(args.graph)
    if args.cuda:
        graph.to('cuda')
    print('Done!')
    
    # partition = GeminiPartition(graph, num_partitions=2, alpha=8 * graph.num_vertices - 1)
    partition = VertexPartition(graph, num_partitions=2)
    cc = ConnectedComponents(graph)
    # strategy = SimpleStrategy(cc)
    strategy = PartitionStrategy(cc, partition)
    
    if args.cuda:
        cc.to('cuda')
    
    t1 = time.time()
    v_data, _ = cc.compute(strategy)
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    if args.cuda:
        os.system("nvidia-smi")
    # output results
    # with open(args.output, 'w') as f:
    #     for i in range(len(cc.vertex_data[:])):
    #         f.write(str(cc.vertex_data[i].item()) + '\n')

if __name__ == '__main__':
    main()
