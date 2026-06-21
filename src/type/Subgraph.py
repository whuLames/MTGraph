import sys
from src.type.Graph import Graph
import torch
import logging

class Subgraph(Graph):
    """
    Subgraph only stores part of the original graph (a part of vertices and their neighbors), but has same return values on those vertices as the original graph.
    """
    def __init__(self, subgraph: Graph, sub_vertices: torch.Tensor, sub_edges: torch.Tensor, 
                 # sub_out_degrees: torch.Tensor, sub_in_degrees: torch.Tensor, 
                 orig_to_sub_vertices: torch.Tensor):
        super().__init__(subgraph.directed)
        self.sub_vertices = sub_vertices
        self.sub_edges = sub_edges
        self.subgraph = subgraph
        
        # stored in CPU, shared between processes
        # self.sub_out_degrees = sub_out_degrees
        # self.sub_in_degrees = sub_in_degrees
        
        # stored as sparse tensor to prevent memory explosion
        self.orig_to_sub_vertices = orig_to_sub_vertices
        
    def orig_to_sub(self, vertices):
        # return self.orig_to_sub_vertices.index_select(0, vertices).to_dense()
        """获取与原图中对应的子图节点集合

        Args:
            vertices: torch.Tensor -- 原图中的节点集合

        Returns:
            vertices: torch.Tensor -- 子图中与之对应的节点集合

        """
        return self.orig_to_sub_vertices[vertices]
    
    @property
    def num_vertices(self):
        return self.sub_vertices.numel()
    
    @property
    def device(self):
        return self.subgraph.device
    
    @property
    def vertices(self):
        return self.sub_vertices
    
    @property
    def num_edges(self):
        return self.sub_edges.numel()
    
    @property
    def edges(self):
        return self.sub_edges
    
    def out_degree(self, vertices):
        # if self.sub_out_degrees is None:
        #     raise NotImplementedError
        # return self.sub_out_degrees[vertices.cpu()].to(self.subgraph.device)
        """查询指定节点出度

        Args:
            vertices: 原图中的节点集合

        Returns:
            out_degree: torch.Tensor -- 先转换为子图中对应节点集, 然后在子图中找对应节点出度
        """
        return self.subgraph.out_degree(self.orig_to_sub(vertices))
    
    def in_degree(self, vertices):
        # if self.sub_in_degrees is None:
        #     raise NotImplementedError
        # return self.sub_in_degrees[vertices.cpu()].to(self.subgraph.device)
        return self.subgraph.in_degree(self.orig_to_sub(vertices))
    
    def out_nbrs(self, vertices):
        return self.subgraph.out_nbrs(self.orig_to_sub(vertices))
        
    def in_nbrs(self, vertices):
        return self.subgraph.in_nbrs(self.orig_to_sub(vertices))
        
    def out_nbrs_csr(self, vertices):
        return self.subgraph.out_nbrs_csr(self.orig_to_sub(vertices))
        
    def all_out_nbrs_csr(self):
        return self.subgraph.all_out_nbrs_csr()
        
    def in_nbrs_csr(self, vertices):
        return self.subgraph.in_nbrs_csr(self.orig_to_sub(vertices))
    
    def all_in_nbrs_csr(self):
        return self.subgraph.all_in_nbrs_csr()
    
    def out_edges(self, vertices):
        return self.subgraph.out_edges(self.orig_to_sub(vertices))
    
    def out_edges_csr(self, vertices):
        return self.subgraph.out_edges_csr(self.orig_to_sub(vertices))
    
    def all_out_edges_csr(self):
        return self.subgraph.all_out_edges_csr()
    
    def in_edges(self, vertices):
        return self.subgraph.in_edges(self.orig_to_sub(vertices))
    
    def in_edges_csr(self, vertices):
        return self.subgraph.in_edges_csr(self.orig_to_sub(vertices))
    
    def all_in_edges_csr(self):
        return self.subgraph.all_in_edges_csr()
    
    def to(self, *args, **kwargs):
        self.sub_vertices = self.sub_vertices.to(*args, **kwargs)
        self.sub_edges = self.sub_edges.to(*args, **kwargs)
        self.subgraph.to(*args, **kwargs)
        # self.sub_out_degrees = self.sub_out_degrees.to(*args, **kwargs)
        # self.sub_in_degrees = self.sub_in_degrees.to(*args, **kwargs)
        # self.orig_to_sub_vertices = self.orig_to_sub_vertices.to_sparse().to(*args, **kwargs)
        self.orig_to_sub_vertices = self.orig_to_sub_vertices.to(*args, **kwargs)
        # self.orig_to_sub_edges = self.orig_to_sub_edges.to(*args, **kwargs)
        
    def pin_memory(self):
        self.sub_vertices = self.sub_vertices.pin_memory()
        self.sub_edges = self.sub_edges.pin_memory()
        self.subgraph.pin_memory()
        # self.sub_out_degrees = self.sub_out_degrees.pin_memory()
        # self.sub_in_degrees = self.sub_in_degrees.pin_memory()
        self.orig_to_sub_vertices = self.orig_to_sub_vertices.pin_memory()
        # self.orig_to_sub_edges = self.orig_to_sub_edges.pin_memory()
        
    def subgraph(self, vertices):
        raise NotImplementedError
    
    def get_vertex_attr(self, vertices, attr):
        return self.subgraph.get_vertex_attr(self.orig_to_sub(vertices), attr)
    
    def select_vertex_by_attr(self, attr, cond):
        return self.subgraph.select_vertex_by_attr(attr, cond)
    
    def set_vertex_attr(self, vertices, attr, value, mask):
        self.subgraph.set_vertex_attr(self.orig_to_sub(vertices), attr, value, mask)
    
    def get_edge_attr(self, edges, attr):
        # return self.subgraph.get_edge_attr(self.orig_to_sub_edges[edges], attr)
        raise NotImplementedError
    
    def select_edge_by_attr(self, attr, cond):
        return self.subgraph.select_edge_by_attr(attr, cond)
    
    def set_edge_attr(self, edges, attr, value, mask):
        # self.subgraph.set_edge_attr(self.orig_to_sub_edges[edges], attr, value, mask)
        raise NotImplementedError
    
    def csr_subgraph(self, vertices: torch.Tensor):
        return self.subgraph.csr_subgraph(self.orig_to_sub(vertices))
    