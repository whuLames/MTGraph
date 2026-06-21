"""
Implements a partition strategy that partitions the graph based on vertices in a way similar to the Gemini framework.
"""
from .Partition import Partition
from src.type.Graph import Graph
import torch
import torch.multiprocessing as mp
from ..helper import divide_equally
import logging 
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

class GeminiPartition(Partition):
    def __init__(self, graph: Graph, num_partitions: int, alpha):
        super().__init__(graph)
        self.n_partitions = num_partitions
        self.vertices = self.graph.vertices
        self.degrees = self.graph.out_degree(self.vertices) + alpha
        self.num = 0
    @property
    def num_partitions(self):
        return self.n_partitions
    
    def generate_partitions(self):
        self.num += 1
        logging.info('Generating partitions for the {}th time'.format(self.num))
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
        