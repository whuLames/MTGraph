"""
Implementation of the framework of the GAS (Gather-Apply-Scatter) structure.
APIs are modeled after the PowerGraph framework.
"""
import sys
import abc
import torch
from src.type.Graph import Graph
import random
from .strategy import Strategy, SimpleStrategy

class GASProgram(abc.ABC):
    def __init__(self, graph: Graph, vertex_data_type=torch.float32, edge_data_type=torch.float32, vertex_data=None, edge_data=None, start_from=None, num_iter = 0, nbr_update_freq=0):
        """
        Initialize a GASProgram object. Provides an interface for the GAS structure. Since PyTorch custom data types are hard to implement, users can specify the data shape to support multiple data. (more stress on the user side)
        
        :param Graph graph: graph to be processed
        :param nbr_update_freq: frequency for the update of gather_nbrs and scatter_nbrs. 0 means no update. 1 means update every iteration. 2 means update every other iteration. etc.

        :param start_from torch.Tensor -- 初始计算节点
        """
        super().__init__()
        self.graph = graph
        self.vertex_data_type = vertex_data_type
        self.edge_data_type = edge_data_type
        # 如果 `start_from` 没有被初始化, 则其值为图中所有节点,即初始状态所有节点参与运算
        if start_from is None:
            start_from = self.graph.vertices
        self.activated_vertex_mask = torch.zeros(graph.num_vertices, dtype=torch.bool)
        # 初始激活节点为 `start_from` 所指定的节点
        self.activated_vertex_mask[start_from] = 1
        # 下一轮次的激活节点掩码, True代表该节点为激活节点
        self.next_activated_vertex_mask = torch.zeros(graph.num_vertices, dtype=torch.bool)
        if vertex_data is None:
            self.vertex_data = torch.zeros(self.graph.num_vertices, dtype=self.vertex_data_type)
        else:
            assert isinstance(vertex_data, torch.Tensor), "vertex_data_shape must be a Tmnsor."
            assert vertex_data.shape[0] == self.graph.num_vertices
            self.vertex_data = vertex_data
            
        if edge_data is None:
            self.edge_data = torch.zeros(self.graph.num_edges, dtype=self.edge_data_type)
        else:
            assert isinstance(edge_data, torch.Tensor), "edge_data must be a Tensor."
            assert edge_data.shape[0] == self.graph.num_edges
            self.edge_data = edge_data
            
        self.nbr_update_freq = nbr_update_freq
        self.curr_iter = 0  # curr_iter: 当前迭代轮次
        self.not_change_activated = False  # 激活节点是否没有改变, False: 激活节点需要更新, True: 激活节点不需要更新
        self.is_quit = False  # 是否退出迭代, True: 退出迭代, False: 继续迭代

    @abc.abstractmethod
    def gather(self, vertices, nbrs, edges, ptr):
        """
        Gather information from the neighbors of vertex_u.
        
        :param Tensor<N> vertices: vertices to gather information from
        :param Tensor<M> nbrs: neighbors of vertices
        :param Tensor<M> edges: edges between vertices and nbrs
        :param Tensor<N+1> ptr: nbrs and edges are arranged in CSR order, where ptr is the pointer
        :return: Tensor<M, d> or Tensor<M> gathered data, Tensor<N+1> CSR pointer
        """
        pass
    
    @abc.abstractmethod
    def sum(self, gathered_data, ptr):
        """
        Sum the gathered information.
        
        :param Tensor<M, d> or Tensor<M> gathered_data: Tensor of gathered information
        :param Tensor<N+1> ptr: CSR pointer
        :return: Tensor<N> or Tensor<N, d> gathered sum
        """
        pass
    
    @abc.abstractmethod
    def apply(self, vertices, gathered_sum):
        """
        Apply the gathered information to vertices.
        
        :param Tensor<N> vertices: vertices to apply information to
        :param Tensor<N> or Tensor<N, d>: gathered sum
        :return: Tensor<N> or Tensor<N, d> or None new data for vertex_u, later to be applied to the vertex, Tensor<N> or Tensor<N, d> or None mask
        """
        pass
    
    @abc.abstractmethod
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        """
        Scatter the gathered information to the neighbors of vertices.
        
        :param Tensor<N> vertices: vertices to scatter information to
        :param Tensor<M> nbrs: neighbors of vertices
        :param Tensor<M> edges: edges between vertices and nbrs
        :param Tensor<N+1> ptr: CSR-style pointer for vertices and nbrs
        :param Tensor<N> or Tensor<N, d> apply_data: results from apply (not found a use case yet)
        :return: Tensor<M> or Tensor<M, d> or None new data, later to be applied to the edges; Tensor<M> or None mask
        """
        pass
    
    @abc.abstractmethod
    def gather_nbrs(self, vertices):
        """
        The neighbors for gathering. Users may specify this to be in-neighbors, out-neighbors, or others.
        
        :param Tensor<N> vertices: vertices
        :return: Tensor<M> neighbors of the vertex, Tensor<M> related edges, Tensor<N+1> CSR pointer
        """
        pass
    
    @abc.abstractmethod
    def scatter_nbrs(self, vertex):
        """
        The neighbors for scattering. Users may specify this to be in-neighbors, out-neighbors, or others.
        
        :param Tensor<N> vertices: vertices
        :return: Tensor<M> neighbors of the vertex, Tensor<M> related edges, Tensor<N+1> CSR pointer
        """
        pass
    
    @property
    def device(self):
        """
        return the device where the vertex/edge data reside.
        :return: device
        """
        assert self.vertex_data.device == self.edge_data.device, "Vertex/edge data is not on the same device."
        return self.vertex_data.device
    
    def to(self, *args, **kwargs):
        """
        Move the vertex/edge data to the specified device.
        
        :return: None
        """
        self.vertex_data = self.vertex_data.to(*args, **kwargs)
        self.edge_data = self.edge_data.to(*args, **kwargs)
        self.activated_vertex_mask = self.activated_vertex_mask.to(*args, **kwargs)
        self.next_activated_vertex_mask = self.next_activated_vertex_mask.to(*args, **kwargs)
        
    def activate(self, vertices):
        """
        Activate a vertex in the GAS model, so pyththat it is put in the queue of vertices to be processed.
        
        :param vertices: vertex to activate
        :return: None
        """
        self.next_activated_vertex_mask[vertices] = 1
        
    def not_change_activated_next_iter(self):
        """
        (For cases nbr_update_freq == 0) For optimization, you can specify not to change the activated vertices next iteration to avoid repetitive computation.
        """
        self.not_change_activated = True
         
    def compute(self, strategy):
        """
        Run computation on the graph and data.
        
        :return: None. check the vertex/edge data after computation.
        """
        # check basic assumptions
        assert self.graph.num_vertices == self.vertex_data.shape[0], "Number of vertices in graph and vertex data do not match."
        
        return strategy.compute()
        
    def set_graph(self, graph):
        self.graph = graph
            