"""
Implementation of SSSP using the GAS framework. Assume that your graph is undirected.
"""

import sys
from src.framework.GASProgram import GASProgram
from src.type.CSRGraph import CSRGraph
from src.type.CSRCGraph import CSRCGraph
from torch_scatter import segment_csr
import torch
import argparse
import time
import numpy as np
import os
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
from src.framework.partition.VertexPartition import VertexPartition
from src.framework.strategy.PartitionStrategy import PartitionStrategy

class SSSP(GASProgram):
    def __init__(self, graph: CSRGraph, source: int, edge_data: torch.Tensor, vertex_data_type=torch.float32):
        # vertex_data: [num_vertices]. stores dists. init to inf (source is 0)
        vertex_data = torch.ones((graph.num_vertices,), dtype=vertex_data_type) * torch.inf
        vertex_data[source] = 0.0 
        self.changed = None
        self.source = source
        super().__init__(graph, vertex_data_type, edge_data.dtype, vertex_data, edge_data)
    
    def gather(self, vertices, nbrs, edges, ptr):
        return self.vertex_data[nbrs] + self.edge_data[edges], ptr
    
    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='min')
    
    def apply(self, vertices, gathered_sum):
        self.changed = self.vertex_data[vertices] > gathered_sum
        if torch.all(self.vertex_data[vertices] <= gathered_sum):
            return None, None
        gathered_sum = torch.min(torch.stack([self.vertex_data[vertices], gathered_sum], dim=0), dim=0)[0]
        return gathered_sum, torch.ones_like(gathered_sum, dtype=torch.bool)

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if self.nbr_update_freq > 0:
            if torch.sum(self.changed) > 0:
                changed_vertices = vertices[self.changed]
                activate_vertices, _ = self.graph.out_nbrs_csr(changed_vertices)
                self.activate(activate_vertices)
            elif self.source in vertices:
                self.activate(self.graph.out_nbrs_csr(vertices)[0])
        else:
            if torch.sum(self.changed) > 0:
                self.not_change_activated_next_iter()
        return None, None
        
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
            out_edges, _ = self.graph.all_out_edges_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
            out_edges, _ = self.graph.out_edges_csr(vertices)
        return out_nbrs, out_edges, ptr
        
    def scatter_nbrs(self, vertices):
        return None, None, None
    
def main():
    # Assume that your graph is undirected
    # Assume that one weight exists for each edge
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--source', type=int, help='source vertex', required=True)
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph, _ = CSRGraph.read_graph(args.graph, split=None, edge_attrs_list=['weight'])
    # graph = CSRCGraph.read_csrc_graph_bin(args.graph)
    if args.cuda:
        graph.to('cuda')
        print(graph.vertices.device)
    data = graph.edge_attrs_tensor
    if len(data) != 1:
        print('No weights provided or too many weights provided. Default to dist(e)=1')
        data = torch.ones((graph.num_edges,), dtype=torch.float32)
    else:
        data = data[0]
    print('Done!')
    # partition = GeminiPartition(graph, num_partitions=2, alpha=8 * graph.num_vertices - 1)
    partition = VertexPartition(graph, num_partitions=2)
    sssp = SSSP(graph, args.source, data)
    # strategy = SimpleStrategy(sssp)
    strategy = SimpleStrategy(sssp)
    
    if args.cuda:
        sssp.to('cuda')
    
    t1 = time.time()
    sssp.compute(strategy)
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(sssp.vertex_data[:])):
            f.write(str(sssp.vertex_data[i].item()) + '\n')
    

if __name__ == '__main__':
    main()
    