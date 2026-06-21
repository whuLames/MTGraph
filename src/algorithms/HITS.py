"""
HITS algorithm implementation with the GAS framework. Assume that the graph is directed.
"""

from ..framework.GASProgram import GASProgram
from ..type.CSRCGraph import CSRCGraph
from ..framework.strategy.SimpleStrategy import SimpleStrategy
import torch
import numpy as np
import argparse
import time
from torch_scatter import segment_csr
from ..framework.helper import batched_csr_selection

class HITS(GASProgram):
    def __init__(self, max_steps: int, graph: CSRCGraph, vertex_data_type=torch.float32, vertex_data=None, num_iter=50):
        # vertex data: [num_vertices, 4]. [:, 0] is hub; [:, 1] is authority. [:. 2:] is delta.
        if vertex_data is None:
            vertex_data = torch.ones((graph.num_vertices, 4), dtype=vertex_data_type)
            vertex_data /= graph.num_vertices  # normalization
        self.max_steps = max_steps
        self.curr_steps = 0
        self.UPDATE_THRESHOLD = 0.0001
        self.ITER_THRESHOLD = num_iter
        super().__init__(graph=graph, vertex_data_type=vertex_data_type, vertex_data=vertex_data)
    
    def gather(self, vertices, nbrs, edges, ptr):
        out_nbrs, in_nbrs = nbrs
        hub_values = self.vertex_data[out_nbrs, 1]
        auth_values = self.vertex_data[in_nbrs, 0]
        return (hub_values, auth_values), ptr
    
    def sum(self, gathered_data, ptr):
        out_ptr, in_ptr = ptr
        hub_values, auth_values = gathered_data
        hub_sum = segment_csr(hub_values, out_ptr, reduce='sum')
        auth_sum = segment_csr(auth_values, in_ptr, reduce='sum')
        return hub_sum, auth_sum
    
    def apply(self, vertices, gathered_sum):
        all_hub_sum = torch.sqrt(torch.sum(self.vertex_data[:, 0] ** 2) - torch.sum(self.vertex_data[vertices, 0] ** 2) + torch.sum(gathered_sum[0] ** 2))
        all_auth_sum = torch.sqrt(torch.sum(self.vertex_data[:, 1] ** 2) - torch.sum(self.vertex_data[vertices, 1] ** 2) + torch.sum(gathered_sum[1] ** 2))
        new_hub = gathered_sum[0] / all_hub_sum
        new_auth = gathered_sum[1] / all_auth_sum
        delta_hub = abs(new_hub - self.vertex_data[vertices, 0])
        delta_auth = abs(new_auth - self.vertex_data[vertices, 1])
        return torch.stack([new_hub, new_auth, delta_hub, delta_auth]), torch.ones((vertices.shape[0], 4), dtype=torch.bool)
            
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if self.nbr_update_freq > 0:
            hub_delta = self.vertex_data[vertices, 2]
            auth_delta = self.vertex_data[vertices, 3]
            hub_selected = torch.where(hub_delta > self.UPDATE_THRESHOLD)[0]
            auth_selected = torch.where(auth_delta > self.UPDATE_THRESHOLD)[0]
            selected = torch.cat([hub_selected, auth_selected])
            if selected.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD:
                # get neighbors of selected
                starts, ends = ptr[selected], ptr[selected + 1]
                result, ptr = batched_csr_selection(starts, ends)
                all_neighbors = nbrs[result]
                self.activate(all_neighbors)
        else:
            if self.curr_iter < self.ITER_THRESHOLD:
                self.not_change_activated_next_iter()
        return None, None
    
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, out_ptr = self.graph.all_out_nbrs_csr()
            in_nbrs, in_ptr = self.graph.all_in_nbrs_csr()
        else:
            out_nbrs, out_ptr = self.graph.out_nbrs_csr(vertices)
            in_nbrs, in_ptr = self.graph.in_nbrs_csr(vertices)
        return (out_nbrs, in_nbrs), None, (out_ptr, in_ptr)
    
    def scatter_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, out_ptr = self.graph.all_out_nbrs_csr()
            in_nbrs, in_ptr = self.graph.all_in_nbrs_csr()
        else:
            out_nbrs, out_ptr = self.graph.out_nbrs_csr(vertices)
            in_nbrs, in_ptr = self.graph.in_nbrs_csr(vertices)
        return (out_nbrs, in_nbrs), None, (out_ptr, in_ptr)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--maxsteps', type=int, help='max steps in iteration')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph = CSRCGraph.read_csrc_graph_bin(args.graph)
    print('Done!')
    
    model = HITS(max_steps=args.maxsteps, graph=graph)
    strategy = SimpleStrategy(model)

    t1 = time.time()
    model.compute(strategy)
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(model.vertex_data[:])):
            f.write(str(model.vertex_data[i, 0].item()) + '\t' + str(model.vertex_data[i, 1].item()) + '\n')

if __name__ == '__main__':
    main()
