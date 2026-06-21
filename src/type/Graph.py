"""
Abstract interface for the Graph data type.
"""

import abc
import numpy as np
import torch

class Graph(abc.ABC):
    def __init__(self, directed=False):
        self.directed = directed
        self.vertices_t, self.edges_t = None, None
        
    @property
    @abc.abstractmethod
    def num_vertices(self):
        pass
    
    @property
    def vertices(self):
        if self.vertices_t is None:
            self.vertices_t = torch.arange(self.num_vertices, device=self.device)
        return self.vertices_t
    
    @property
    @abc.abstractmethod
    def num_edges(self):
        pass
    
    @property
    def edges(self):
        if self.edges_t is None:
            self.edges_t = torch.arange(self.num_edges, device=self.device)
        return self.edges_t
    
    @abc.abstractmethod
    def out_degree(self, vertices):
        pass
    
    @abc.abstractmethod
    def in_degree(self, vertices):
        pass
    
    def all_degree(self, vertices):
        if not self.directed:
            return self.out_degree(vertices)
        return self.out_degree(vertices) + self.in_degree(vertices)
    
    @abc.abstractmethod
    def out_nbrs(self, vertices):
        pass
    
    @abc.abstractmethod
    def in_nbrs(self, vertices):
        pass
    
    def all_nbrs(self, vertices):
        if not self.directed:
            return self.out_nbrs(vertices)
        out_n, out_n_mask = self.out_nbrs(vertices)
        in_n, in_n_mask = self.in_nbrs(vertices)
        return torch.cat((out_n, in_n), dim=1), torch.cat((out_n_mask, in_n_mask), dim=1)
    
    @abc.abstractmethod
    def out_nbrs_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_out_nbrs_csr(self):
        pass
    
    @abc.abstractmethod
    def in_nbrs_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_in_nbrs_csr(self):
        pass
    
    def all_nbrs_csr(self, vertices):
        if not self.directed:
            return self.out_nbrs_csr(vertices)
        out_n, out_n_ptr = self.out_nbrs_csr(vertices)
        in_n, in_n_ptr = self.in_nbrs_csr(vertices)
        ptr = out_n_ptr + in_n_ptr
        nbrs = torch.zeros((out_n.shape[0] + in_n.shape[0]), dtype=out_n.dtype, device=out_n.device)
        curr_beg = 0
        for i in range(1, len(ptr)+1):
            curr_end = curr_beg + out_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[out_n_ptr[i-1]:out_n_ptr[i]]
            curr_beg = curr_end
            curr_end = curr_beg + in_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[in_n_ptr[i-1]:in_n_ptr[i]]
            curr_beg = curr_end
        return nbrs, ptr
    
    @abc.abstractmethod
    def out_edges(self, vertices):
        pass
    
    @abc.abstractmethod
    def in_edges(self, vertices):
        pass
    
    def all_edges(self, vertices):
        if not self.directed:
            return self.out_edges(vertices)
        out_e, out_e_mask = self.out_edges(vertices)
        in_e, in_e_mask = self.in_edges(vertices)
        return torch.cat((out_e, in_e), dim=1), torch.cat((out_e_mask, in_e_mask), dim=1)
    
    @abc.abstractmethod
    def out_edges_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_out_edges_csr(self):
        pass
    
    @abc.abstractmethod
    def in_edges_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_in_edges_csr(self):
        pass
    
    def all_edges_csr(self, vertices):
        if not self.directed:
            return self.out_edges_csr(vertices)
        out_n, out_n_ptr = self.out_edges_csr(vertices)
        in_n, in_n_ptr = self.in_edges_csr(vertices)
        ptr = out_n_ptr + in_n_ptr
        nbrs = torch.zeros((out_n.shape[0] + in_n.shape[0]), dtype=out_n.dtype, device=out_n.device)
        curr_beg = 0
        for i in range(1, len(ptr)+1):
            curr_end = curr_beg + out_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[out_n_ptr[i-1]:out_n_ptr[i]]
            curr_beg = curr_end
            curr_end = curr_beg + in_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[in_n_ptr[i-1]:in_n_ptr[i]]
            curr_beg = curr_end
        return nbrs, ptr
    
    @abc.abstractmethod
    def device(self):
        pass
    
    @abc.abstractmethod
    def to(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def pin_memory(self):
        pass
    
    @abc.abstractmethod
    def subgraph(self, vertices):
        """
        Induced subgraph from vertices.
        """
        pass
    
    @abc.abstractmethod
    def get_vertex_attr(self, vertices, attr):
        pass
    
    @abc.abstractmethod
    def select_vertex_by_attr(self, attr, cond):
        pass
    
    @abc.abstractmethod
    def set_vertex_attr(self, vertices, attr, value, mask):
        pass
    
    @abc.abstractmethod
    def get_edge_attr(self, edges, attr):
        pass
    
    @abc.abstractmethod
    def select_edge_by_attr(self, attr, cond):
        pass
    
    @abc.abstractmethod
    def set_edge_attr(self, edges, attr, value, mask):
        pass
    
    @abc.abstractmethod
    def csr_subgraph(self, vertices: torch.Tensor):
        pass
    
    def read_edgelist(f, split=None):
        """
        Read edge-list from a file. Allow one value for each edge.
        
        :param f: file to read from
        :param str split: split string, such as spaces or tabs.
        :return: edge_starts, edge_ends, vertices, edge_data (a list of np.arrays, each is a column)
        """
        print('-------- {} ------------'.format(f))
        array = np.loadtxt(f, dtype=np.int32)
        # 将 edge_starts, edge_ends 进行排序
        sort_indices = np.lexsort((array[:, 1], array[:, 0]))
        array = array[sort_indices]
        edge_starts = array[:, 0]
        edge_ends = array[:, 1]
        data = array[:, 2:].T
        vertices = np.unique(np.sort(np.concatenate((edge_starts, edge_ends))))
        
        return edge_starts, edge_ends, vertices, data
