"""
Implements a partition strategy that partitions the graph based on vertices, but with average # of edges.
"""

from .Partition import Partition
from ...type.Graph import Graph
import torch
import torch.multiprocessing as mp
from ..helper import divide_equally

class EdgePartition(Partition):
    def __init__(self, graph: Graph, num_partitions: int):
        super().__init__(graph)
        self.n_partitions = num_partitions
        self.vertices = self.graph.vertices
        self.degrees = self.graph.out_degree(self.vertices)
    
    @property
    def num_partitions(self):
        return self.n_partitions
    
    def generate_partitions(self):
        results = []
        new_vertices = [[] for _ in range(self.num_partitions)]
        indices, _ = divide_equally(self.degrees, self.num_partitions)
        for i in range(len(indices)):
            new_vertices[i] = self.vertices[indices[i]]
        for vs in new_vertices:
            result = (self.graph.csr_subgraph(vs), vs)
            results.append(result)
        return results
    
    def set_graph(self, graph: Graph):
        self.graph = graph
        self.vertices = self.graph.vertices
        self.degrees = self.graph.out_degree(self.vertices)
        
    def set_num_partitions(self, n_partitions):
        self.n_partitions = n_partitions
        