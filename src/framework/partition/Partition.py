"""
Partition.py provides an interface for partitioning a graph into different subgraphs according to GPU memory, specific strategies, etc.
"""

import abc
from src.type.Graph import Graph

class Partition(abc.ABC):
    def __init__(self, graph: Graph):
        self.graph = graph
        super().__init__()

    @property
    @abc.abstractmethod
    def num_partitions(self):
        pass
    
    @abc.abstractmethod
    def generate_partitions(self):
        pass
    
    @abc.abstractmethod
    def set_graph(self, graph: Graph):
        pass
    
    @abc.abstractmethod
    def set_num_partitions(self, n_partitions):
        pass
