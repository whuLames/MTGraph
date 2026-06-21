"""
Implementation of the BFS algorithm using the TCR. Yield the edges of graphs in a BFS order. Distance is stored in vertex_data. Well-defined for directed graphs.
"""
import sys
import logging
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.framework.strategy.SimpleStrategy import SimpleStrategy
import torch
import argparse
import time
from torch_scatter import segment_csr
LONG_MAX = 100000000000
class BFS(GASProgram):
    def __init__(self, graph: CSRGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, vertex_data=None, edge_data=None, start_from=None):
        # no need for edge_data
        # vertex_data: [num_vertices]. stores dists. init to inf (source is 0)
        if vertex_data is None:
            vertex_data = torch.ones((graph.num_vertices,), dtype=vertex_data_type) * torch.inf
        vertex_data[start_from] = 0
        # start_from = None
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data, start_from=None)
        # logging.info('vertex_data: {}'.format(self.vertex_data))
        # records searched vertices
        self.searched = torch.BoolTensor(graph.num_vertices, device=self.device).fill_(False)
        self.searched[start_from] = True
        self.last_searched = 1
        
    def gather(self, vertices, nbrs, edges, ptr):
        # logging.info('vertices: {}'.format(vertices))
        
        # 将对应节点位置设置为 True
        # self.searched = self.searched.scatter(0, vertices, True)
        # logging.info('self.searched: {}'.format(self.searched))
        # logging.info('vertex_data: {}'.format(self.vertex_data[nbrs] + 1))
        return self.vertex_data[nbrs] + 1, ptr
    
    def sum(self, gathered_data, ptr):
        # logging.info('gathered_data: {}'.format(gathered_data))
        # if torch.any(gathered_data == 0):
        #     logging.info('gathered_data == 0')
        # else:
        #     logging.info('gathered_data != 0')
        # logging.info('ptr: {}'.format(ptr))
        gsum = segment_csr(gathered_data, ptr, reduce='min')

        # 将 0 元素 填充为 Long_MAX 表示不可达
        # gsum = torch.where(gsum == 0, torch.tensor(LONG_MAX, dtype=torch.int64, device=self.device), gsum)
        # mask = torch.eq(gsum, 0)
        # mask[0] = False  # 去除源节点 需要后续修改 因为不是所有测试中 source 都是 0
        # searched_vertices = torch.masked_select(self.graph.vertices, mask)
        # self.searched = self.searched.scatter(0, searched_vertices, True)
        # gsum = torch.where(gsum == 0, torch.tensor(LONG_MAX, dtype=torch.int64, device=self.device), gsum)
        gsum = torch.where(gsum == 3.4028234663852886e+38, torch.tensor(torch.inf, dtype=torch.float32, device=self.device), gsum)
        mask = torch.eq(gsum, 0)
        mask[0] = False
        unreach_vertices = torch.masked_select(self.graph.vertices, mask)
        # 将 unreachable 的节点的搜索状态设置为 True
        self.searched = self.searched.scatter(0, unreach_vertices, True)
        gsum = torch.where(gsum == 0, torch.tensor(torch.inf, dtype=torch.float32, device=self.device), gsum)
        # gsum[0] = 0.0
        return gsum
    
    def apply(self, vertices, gathered_sum):
        """
        Returns:
            apply_data: 
            apply_mask:
        """
        # logging.info('gathered_sum: {}'.format(gathered_sum))
        # if torch.any(gathered_sum == 0):
        #     logging.info('gathered_sum == 0')
        # else:
        #     logging.info('gathered_sum != 0')
        # logging.info('vertices: {}'.format(vertices))
        # logging.info('vertex_data: {}'.format(self.vertex_data[vertices]))
        mask = self.vertex_data[vertices] > gathered_sum[vertices]
        # logging.info('mask: {}'.format(mask))
        # True means it is need to update the vertex_data
        searched_vertices = torch.masked_select(vertices, mask)
        # logging.info('searched_vertices: {}'.format(searched_vertices))
        self.searched = self.searched.scatter(0, searched_vertices, True)
        # logging.info('self.searched num: {}'.format(torch.sum(self.searched).item()))
        # logging.info('self.searched: {}'.format(self.searched))
        return gathered_sum[vertices], mask
    
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        # logging.info('vertices: {}'.format(vertices))
        # not_in_searched = ~self.searched[nbrs]
        # logging.info('self.searched: {}'.format(self.searched))
        not_in_searched = torch.logical_not(self.searched[vertices])
        # logging.info('not_in_searched: {}'.format(not_in_searched))
        now_searched = torch.sum(self.searched).item()
            
        # if self.nbr_update_freq == 0:
        #     if torch.all(not_in_searched == False):
        #         self.not_change_activated_next_iter()
        #         logging.info('not_change_activated_next_iter')
        #     else:
        #         # 激活节点的邻居节点中有未被搜索过的节点, 则将其加入到下一轮次的激活节点中
        #         self.activate(torch.masked_select(nbrs, not_in_searched))
        # else:
        #     self.activate(torch.masked_select(nbrs, not_in_searched))
        # self.activate(torch.masked_select(nbrs, not_in_searched))
        if now_searched != self.last_searched and \
           self.nbr_update_freq == 0 and \
           not torch.all(not_in_searched == False):
            # logging.info('vetex_data: {}'.format(self.vertex_data))
            # self.activate(torch.masked_select(vertices, not_in_searched))
            self.not_change_activated_next_iter()
            self.last_searched = now_searched
        return None, None
    
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            in_nbrs, ptr = self.graph.all_in_nbrs_csr()
        else:
            in_nbrs, ptr = self.graph.in_nbrs_csr(vertices)
        return in_nbrs, None, ptr
        # in_nbrs, ptr = self.graph.in_nbrs_csr(vertices)
        # return in_nbrs, None, ptr
    
    def scatter_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr
        # in_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        # return in_nbrs, None, ptr

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
    # graph, _,data = CSRCGraph.read_graph(args.graph, split=None)
    graph = CSRCGraph.read_csrc_graph_bin(args.graph)
    in_degree = graph.in_degree(graph.vertices)
    mask = torch.eq(in_degree, 0)
    sum = torch.sum(mask).item()
    print('Done!')
    bfs = BFS(graph, start_from=[args.source])
    strategy = SimpleStrategy(bfs)

    if args.cuda:
        logging.info('Using cuda...')
        graph.to('cuda')
        bfs.to('cuda')
        bfs.searched = bfs.searched.to('cuda')
        # graph.vertices = graph.vertices.to('cuda')
    
    t1 = time.time()
    bfs.compute(strategy)
    torch.cuda.synchronize()
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(bfs.vertex_data[:])):
            f.write(str(bfs.vertex_data[i].item()) + '\n')

if __name__ == '__main__':
    main()
