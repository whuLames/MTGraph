"""
Implements a simple vertex-centric partition strategy based on vertex ID.
"""
import sys


from .Partition import Partition
from src.type.Graph import Graph
import torch
import torch.multiprocessing as mp
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class VertexPartition(Partition):
    def __init__(self, graph: Graph, num_partitions: int):
        super().__init__(graph)
        self.n_partitions = num_partitions
        self.partition_size = graph.num_vertices // num_partitions
        self.partition_remainder = graph.num_vertices % num_partitions
        self.num = 0
    @property
    def num_partitions(self):
        return self.n_partitions
    
    # returns (subgraph, indices), vertices
    def generate_partition(self, id):
        beg = id * self.partition_size
        end = beg + self.partition_size
        if id == self.num_partitions - 1:
            end += self.partition_remainder
        new_vertices = self.graph.vertices[beg:end]
        return self.graph.csr_subgraph(new_vertices), torch.arange(beg, end)
    
    def generate_partitions(self):
        self.num += 1
        logging.info('Generating partitions for the {}th time'.format(self.num))
        partitions = list(map(self.generate_partition, range(self.num_partitions)))
        return partitions
    
    def set_graph(self, graph: Graph):
        self.graph = graph
        self.partition_size = graph.num_vertices // self.num_partitions
        self.partition_remainder = graph.num_vertices % self.num_partitions
        
    def set_num_partitions(self, n_partitions):
        self.n_partitions = n_partitions
        self.partition_size = self.graph.num_vertices // self.num_partitions
        self.partition_remainder = self.graph.num_vertices % self.num_partitions
        