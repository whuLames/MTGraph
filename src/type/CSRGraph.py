"""
A Graph type implemented with CSR (compressed sparse row) type.
"""
import torch
import numpy as np
from .Graph import Graph
from torch.nn.utils.rnn import pad_sequence
from src.framework.helper import batched_csr_selection, batched_adj_selection

class CSRGraph(Graph):
    """
    CSR implementation of Graph. Provides efficient access to out_nbrs.
    """
    def __init__(self,
                 columns: torch.Tensor=None, 
                 row_ptr: torch.Tensor=None, 
                 directed=False,
                 vertex_attrs_list=[],
                 vertex_attrs_tensor: torch.Tensor=None,
                 vertex_attrs_mask: torch.Tensor=None,
                 edge_attrs_list=[],
                 edge_attrs_tensor: torch.Tensor=None,
                 edge_attrs_mask: torch.Tensor=None):
        """
        Initialize a CSRGraph object with according datatypes (tensors).
        
        :param Tensor columns: out-neighbors of vertex (arranged in order)
        :param Tensor row_ptr: pointers of each vertex for val and col_ind
        :param bool directed: whether the graph is directed
        :param list vertex_attrs_list: list of vertex attributes names
        :param Tensor vertex_attrs_tensor: tensor of vertex attributes that stores data
        :param Tensor vertex_attrs_mask: mask of vertex attributes
        :param list edge_attrs_list: list of edge attributes names
        :param Tensor edge_attrs_tensor: tensor of edge attributes that stores data
        :param Tensor edge_attrs_mask: mask of edge attributes
        :return: None
        """
        super().__init__(directed=directed)
        self.columns = columns
        self.row_ptr = row_ptr
        self.out_degrees = torch.diff(self.row_ptr)
        # process attributes
        self.vertex_attrs_list = vertex_attrs_list
        self.vertex_attrs_map = {attr: i for i, attr in enumerate(vertex_attrs_list)}
        self.edge_attrs_list = edge_attrs_list
        self.edge_attrs_map = {attr: i for i, attr in enumerate(edge_attrs_list)}
        if vertex_attrs_tensor is not None and vertex_attrs_mask is not None:
            self.vertex_attrs_tensor = vertex_attrs_tensor
            self.vertex_attrs_mask = vertex_attrs_mask
        else:
            self.vertex_attrs_tensor = torch.zeros((self.num_vertices, len(vertex_attrs_list)), dtype=torch.float32)
            self.vertex_attrs_mask = torch.zeros((self.num_vertices, len(vertex_attrs_list)), dtype=torch.bool)
        if edge_attrs_tensor is not None and edge_attrs_mask is not None:
            self.edge_attrs_tensor = edge_attrs_tensor
            self.edge_attrs_mask = edge_attrs_mask
        else:
            self.edge_attrs_tensor = torch.zeros((self.num_edges, len(edge_attrs_list)), dtype=torch.float32)
            self.edge_attrs_mask = torch.zeros((self.num_edges, len(edge_attrs_list)), dtype=torch.bool)
            
    @property
    def num_vertices(self):
        """返回Graph的节点数目

        Returns:
            (int): the num of vertices

        """
        if hasattr(self.row_ptr, 'shape'):
            return self.row_ptr.shape[0] - 1
        else:
            return 0
        
    @property
    def num_edges(self):
        """number of edges."""
        if hasattr(self.columns, 'shape'):
            return self.columns.shape[0]
        else:
            return 0
    
    def out_degree(self, vertices):
        """Get the number of out neighbors. (if undirected, #out_nbrs = #in_nbrs)
        for every vertex in arg "vertices", return their own out_degree

        Args:
            vertices: torch.Tensor(1*n)--需要查询的出度的节点列表

        Returns:
            torch.Tensor（1*n） -- 每个节点对应的出度


        """
        assert torch.all(vertices < self.num_vertices)
        return self.out_degrees[vertices]
    
    def in_degree(self, vertices):
        """获取节点的出度
        CSR存储格式可以快速获取节点的出度，获取入度见CSCGraph
        Args:
            vertices:

        Returns:

        """
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def out_nbrs(self, vertices):
        """返回指定节点的出度邻居

        Args：
            vertices： torch.Tensor(1*n) -- 需要查询出度邻居的节点列表

        Returns:
            result: torch.Tensor(n * m) -- 每个节点对应的出度邻居列表 n:vertices长度  m: graph。num_vertices - 1
            mask: torch.Tensor(n * m) -- 列表掩码，对应位置是邻居为True 否则为False

        Examples:
            graph: [0,1] [0,2] [0,3] [1,2] [1,3] [2,3]
            result: tensor([[ 1,  2,  3],    mask: tensor([[ True,  True,  True],
                           [ 2,  3, -1],                   [ True,  True, False],
                           [ 3, -1, -1]])                  [ True, False, False]])
        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, mask = batched_adj_selection(starts, ends)
        result = torch.where(mask, self.columns[result], torch.ones_like(result) * -1)
        return result, mask
    
    def out_nbrs_csr(self, vertices):
        """返回指定节点的出度邻居
        以CSR 存储的形式返回指定节点的出度邻居, 返回值为 columns, row_ptr

        Args:
            vertices： torch.Tensor(1*n) -- 需要查询出度邻居的节点列表

        Returns:
            result: torch.Tensor -- 对应节点CSR格式下的columns
            prt: torch.Tensor  --  对应节点CSR格式下的row-ptr
        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, ptr = batched_csr_selection(starts, ends)
        result = self.columns[result]
        return result, ptr
    
    def all_out_nbrs_csr(self):
        """返回所有节点的CSR格式的出度邻居

        Args:

        Returns:
            columns: torch.Tensor
            row_ptr: torch.Tensor
        """
        return self.columns, self.row_ptr

    def in_nbrs(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def in_nbrs_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def all_in_nbrs_csr(self):
        raise NotImplementedError('Not implemented for CSRGraph.')
     
    def out_edges(self, vertices):
        """获取指定节点的出边

        Args：
        vertices： torch.Tensor(1*n) -- 需要查询出边的节点列表

        Returns:
            result: torch.Tensor(n * m) -- 每个节点对应的出边列表 n:vertices长度  m: graph。num_vertices - 1
            mask: torch.Tensor(n * m) -- 列表掩码，对应位置是一条出边为 True 否则为 False

        Examples:
            graph: [0,1] [0,2] [0,3] [1,2] [1,3] [2,3]
            result: tensor([[ 0, 1, 2],    mask: tensor([[ True,  True,  True],
                            [ 3, 4, -1],                 [ True,  True, False],
                            [ 5, -1, -1]])               [ True, False, False]])
        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, mask = batched_adj_selection(starts, ends)
        return result, mask
    
    def all_out_edges_csr(self):
        """返回所有节点的出边

        返回以CSR格式表示的所有节点的出边

        Returns:
            columns: torch.Tensor
            row_ptr: torch.Tensor
        """
        return torch.arange(self.num_edges, device=self.device), self.row_ptr
    
    def out_edges_csr(self, vertices):
        """返回指定节点的CSR存储格式
        以CSR 存储的形式返回指定节点的出度邻居, 返回值为 columns, row_ptr

        Args:
            vertices： torch.Tensor(1*n) -- 需要查询出度邻居的节点列表

        Returns:
            result: torch.Tensor -- 对应节点CSR格式下的columns
            prt: torch.Tensor  --  对应节点CSR格式下的row-ptr

        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, ptr = batched_csr_selection(starts, ends)
        return result, ptr
    
    def in_edges(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def in_edges_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def all_in_edges_csr(self):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    @property
    def device(self):
        """
        return the device where the graph resides.
        :return: device
        """
        col_ind_dev = self.columns.device
        row_ind_dev = self.row_ptr.device
        assert col_ind_dev == row_ind_dev, "Graph is not on the same device."
        
        return col_ind_dev
        
    def to(self, *args, **kwargs):
        """
        Move the graph to the specified device.
        
        :return: None
        """
        self.columns = self.columns.to(*args, **kwargs)
        self.row_ptr = self.row_ptr.to(*args, **kwargs)
        self.out_degrees = self.out_degrees.to(*args, **kwargs)
        # check
        if self.vertices_t != None:
            self.vertices_t = self.vertices_t.to(*args, **kwargs)
        
    def pin_memory(self):
        """锁页机制
        锁页机制，将数据指定存储在内存中, 指定os不会将该数据换出到虚拟内存中
        节省数据换入换出时间，提高执行效率

        """
        self.columns = self.columns.pin_memory()
        self.row_ptr = self.row_ptr.pin_memory()
    
    def csr_subgraph(self, vertices: torch.Tensor):
        """获取CSR格式子图
        根据传入的节点集合, 生成子图. 生成的子图中包含节点集合中每一个节点的所有邻居节点，而不仅仅是在节点集合中的邻居节点。
        这样的子图分割方式便于后续的分割计算

        Args:
            vertices: torch.Tensor -- 节点集合

        Returns:
            subgraph: CSRGraph -- 根据节点集合得到的CSR格式的子图
            indices:
        """
        sub_degrees = self.out_degrees[vertices]
        sub_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int64, device=self.device), sub_degrees.cumsum(0)])
        # fetch sub_columns
        # starts, ends = sub_row_ptr[:-1], sub_row_ptr[1:]
        # starts, ends = starts[vertices], ends[vertices]

        # 获取每个节点所对应columns索引的起始位置和终止位置
        starts, ends = self.row_ptr[vertices], self.row_ptr[vertices + 1]
        # size: torch.Tensor
        sizes = (ends - starts)
        # size.sum 即为所有节点边的总数
        ranges = torch.arange(sizes.sum(), device=self.device)
        # 获取 indices subgraph 中的columns到 original graph的columns中的映射
        indices = ranges + starts.repeat_interleave(sizes) - (sub_row_ptr[:-1]).repeat_interleave(sizes)
        sub_columns = self.columns[indices]
        
        # fetch attributes  根据索引映射 indices 获取subgraph中的边属性和点属性
        sub_vertex_attrs_tensor, sub_vertex_attrs_mask = None, None
        if self.vertex_attrs_tensor is not None:
            sub_vertex_attrs_tensor = self.vertex_attrs_tensor[vertices]
            sub_vertex_attrs_mask = self.vertex_attrs_mask[vertices]
        sub_edge_attrs_tensor, sub_edge_attrs_mask = None, None
        if self.edge_attrs_tensor is not None:
            sub_edge_attrs_tensor = self.edge_attrs_tensor[indices]
            sub_edge_attrs_mask = self.edge_attrs_mask[indices]
        
        return CSRGraph(sub_columns, sub_row_ptr, self.directed, self.vertex_attrs_list,
                        sub_vertex_attrs_tensor, sub_vertex_attrs_mask,
                        self.edge_attrs_list, sub_edge_attrs_tensor, sub_edge_attrs_mask), indices
        
    def subgraph(self, vertices: torch.Tensor):
        """

        Args:
            vertices: torch.Tensor -- 子图所含有的节点

        Returns:
            subgraph: CSRGraph -- 根据子图节点生成的CSR格式存储的Graph
            new_vertices_to_od:
        """
        # map
        # new_vertices_to_old = vertices.sort().unique_consecutive()
        # 先去重，再排序
        new_vertices_to_old = vertices.unique_consecutive().sort()[0]
        print(new_vertices_to_old)
        old_vertices_to_new = {}
        for i, v in enumerate(new_vertices_to_old):
            old_vertices_to_new[v] = i
        # to LIL
        all_nbrs = []
        new_nbrs_list = []
        lengths = [0]
        for i in range(len(self.row_ptr) - 1):
            all_nbrs = self.columns[self.row_ptr[i]:self.row_ptr[i+1]]
        # leave specified vertices in LIL
        for nbrs in all_nbrs:
            nbrs = nbrs[torch.where(nbrs in vertices)[0]]
            for i, e in enumerate(nbrs):
                nbrs[i] = old_vertices_to_new[e]
            new_nbrs_list.append(nbrs)
            lengths.append(len(nbrs))
        # LIL to CSR
        new_nbrs = torch.cat(new_nbrs_list)
        ptr = torch.tensor(lengths, dtype=torch.int64, device=self.device).cumsum(0)
        return CSRGraph(new_nbrs, ptr), new_vertices_to_old

    def get_vertex_attr(self, vertices, attr):
        assert torch.all(vertices < self.num_vertices)
        attr_id = self.vertex_attrs_map[attr]
        return self.vertex_attrs[attr_id][vertices]
    
    def select_vertex_by_attr(self, attr, cond):
        attr_id = self.vertex_attrs_map[attr]
        return torch.where(cond(self.vertex_attrs[attr_id]))[0]
    
    def set_vertex_attr(self, vertices, attr, value, mask):
        assert torch.all(vertices < self.num_vertices)
        attr_id = self.vertex_attrs_map[attr]
        self.vertex_attrs[attr_id][vertices] = torch.where(mask, value, self.vertex_attrs[attr_id][vertices])
    
    def get_edge_attr(self, edges, attr):
        assert torch.all(edges < self.num_edges)
        attr_id = self.edge_attrs_map[attr]
        return self.edge_attrs[attr_id][edges]
    
    def select_edge_by_attr(self, attr, cond):
        attr_id = self.edge_attrs_map[attr]
        return torch.where(cond(self.edge_attrs[attr_id]))[0]
    
    def set_edge_attr(self, edges, attr, value, mask):
        assert torch.all(edges < self.num_edges)
        attr_id = self.edge_attrs_map[attr]
        self.edge_attrs[attr_id][edges] = torch.where(mask, value, self.edge_attrs[attr_id][edges])

    @staticmethod   
    def read_csr_graph(f, split=None):
        """
        直接读取CSRC存储格式的文件
        只读取前两行作为CSRGraph数据
        """
        print('-------- {} ------------'.format(f))
        f = open(f, 'r')
        lines = f.readlines()
        info = lines[0].split(' ')
        vertex_cnt = int(info[0])
        edge_cnt = int(info[1])
        print(vertex_cnt, " ", edge_cnt)
        row_ptr = lines[1].split(' ')
        print(len(row_ptr))
        columns = lines[2].split(' ')
        print(len(columns))
        # col_ptr = lines[3].split(' ')
        # print(len(col_ptr))
        # rows = lines[4].split(' ')
        # print(len(rows))
        # print(len(row_ptr), " ", len(columns), " ", len(col_ptr), " ", len(rows))
        shuffle_ptr = None
        # generate csrc graph
        # print(type(row_ptr[0]))
        # print(len(row_ptr), " ", len(columns), " ", len(col_ptr), " ", len(rows))
        row_ptr = [int(i) for i in row_ptr[:-1]]
        columns = [int(i) for i in columns[:-1]]
        # col_ptr = [int(i) for i in col_ptr[:-1]]
        # rows = [int(i) for i in rows[:-1]]
        return CSRGraph(
            columns=columns,
            row_ptr=row_ptr,
            directed=True,
        )
    
    @staticmethod
    def edge_list_to_Graph(edge_starts, edge_ends, directed=False, vertices=None, edge_attrs=None, edge_attrs_list=[], vertex_attrs=None, vertex_attrs_list=[]):
        """将edge_list的图表示形式转换为 CSRGraph格式

        Read edge_lists and return an according CSRGraph.
        
        :param np.array edge_starts: starting points of edges
        :param np.array edge_ends: ending points of edges
        :param bool directed: whether the graph is directed
        :param np.array vertices: vertices. can be None
        :param List[np.array] edge_attrs: a list data for each edge
        :param List edge_attrs_list: a list of edge attributes (preferably strings, like names of the attributes)
        :param List[np.array] vertex_attrs: a list data for each vertex (in the same order as vertices. please don't set vertices=None if you use this)
        :param List vertex_attrs_list: a list of vertex attributes (preferably strings, like names of the attributes)
        :return: CSRGraph, a dictionary of vertex to index, and a list of edge data in Tensor and CSR order

        Returns:
            graph: CSRGraph -- 生成的CSR存储格式的图
            vertex_to_index: dictionary -- map vertex to index
        """
        if vertices is None:
            vertices = np.array([], dtype=np.int64)
            for s, d in zip(edge_starts, edge_ends):
                vertices = np.append(vertices, s)
                vertices = np.append(vertices, d)
            vertices = np.unique(np.sort(vertices))
            
        # get vertex to index mapping
        vertex_to_index = {}
        # vertex_data_list = [[], [], []]
        vertex_data_list = [[] for _ in range(len(vertex_attrs_list))]
        # 建立节点到索引的映射, 索引从0开始
        for vertex, index in zip(vertices, range(len(vertices))):
            vertex_to_index[vertex] = index
            if vertex_attrs is not None:
                for data_index, data in enumerate(vertex_attrs):
                    vertex_data_list[data_index].append(data[vertex])  # maybe it should be:
                    # vertex_data_list[data_index].append(data[index])

        # sort edge lists into val, col_ind, and row_ind
        num_vertices = len(vertices)
        num_data = len(edge_attrs)
        col_ind_list = [[] for _ in range(num_vertices)]
        data_list = [[[] for _ in range(num_vertices)] for _ in range(num_data)]
        for start, end, *data in zip(edge_starts, edge_ends, *edge_attrs):
            start_v = vertex_to_index[start]
            end_v = vertex_to_index[end]
            col_ind_list[start_v].append(end_v)
            if not directed: # undirected
                col_ind_list[end_v].append(start_v)
            for d in range(num_data):
                data_list[d][start_v].append(data[d])
                if not directed:
                    data_list[d][end_v].append(data[d])  
                      
        if not directed:  # unique
            for i in range(len(col_ind_list)):
                col_ind_list[i] = np.unique(col_ind_list[i]).tolist()

        col_ind = torch.zeros(sum([len(l) for l in col_ind_list]), dtype=torch.int64)
        row_ind = torch.zeros(num_vertices + 1, dtype=torch.int64)
        data_tensor = [torch.zeros(sum([len(l) for l in col_ind_list]), dtype=torch.int64) for _ in range(num_data)]
        curr_index = 0
        for l, v, *d in zip(col_ind_list, range(num_vertices), *data_list):
            col_ind[curr_index:curr_index + len(l)] = torch.tensor(l, dtype=torch.int64)
            row_ind[v] = curr_index
            for d2 in range(num_data):
                data_tensor[d2][curr_index:curr_index + len(l)] = torch.tensor(d[d2], dtype=torch.float32)
            curr_index += len(l)
        row_ind[-1] = curr_index
        if len(data_tensor) != 0:
            edge_attrs_tensor = torch.stack(data_tensor, dim=0)
            edge_attrs_mask = torch.ones(edge_attrs_tensor.shape, dtype=torch.bool)
        else:
            edge_attrs_tensor, edge_attrs_mask = None, None
        if vertex_attrs is not None:
            vertex_attrs_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in vertex_data_list], dim=0)
            vertex_attrs_mask = torch.ones(vertex_attrs_tensor.shape, dtype=torch.bool)
        else:
            vertex_attrs_tensor = None
            vertex_attrs_mask = None
        return CSRGraph(col_ind, row_ind, directed, vertex_attrs_list, 
                        vertex_attrs_tensor, vertex_attrs_mask, edge_attrs_list,
                        edge_attrs_tensor, edge_attrs_mask), vertex_to_index
        
    @staticmethod
    def read_graph(f, split=None, directed=False, edge_attrs_list=[]):
        """
        Read an edgelist file and return an according CSRGraph.
        Edge lists should has the following format:
        v_0[split]v_1
        values will default to .0.
        By default, graphs are stored in CPU.
        
        :param str f: filename for edge list
        :param str split: split string for each line
        :param bool directed: whether the graph is directed
        :return: CSRGraph and a dictionary of vertex to index   `
        """
        edge_starts, edge_ends, vertices, data = Graph.read_edgelist(f, split)
        return CSRGraph.edge_list_to_Graph(edge_starts, edge_ends, directed, vertices, edge_attrs=data, edge_attrs_list=edge_attrs_list)

    @staticmethod
    def read_graph_bin(f, is_long=False):
        """
        read csr graph from binary file
        is_long: whether the data type is int64
            一般来讲,节点数量很难超过int32的表示范围,即2147483647
            但是,边的数量可能会超过int32的表示范围
            所以,当边的数量超过int32时, row_ptr的数据类型为应为int64
        """
        # print('read csr graph from binary file : {}'.format(f))

        import os
        path_v = f + '/csr_vlist.bin'
        path_e = f + '/csr_elist.bin'

        # read v_list
        # for twitter dataset, int32 will overflow for row_ptr
        sz = os.path.getsize(path_v)
        sz = sz // 4 if not is_long else sz // 8
        # print('row_ptr sz {}'.format(sz))
        data_type = np.int32 if not is_long else np.int64
        row_ptr = np.fromfile(path_v, dtype=data_type, count=sz)
        row_ptr = torch.from_numpy(row_ptr).to(torch.int64)

        # read e_list
        sz = os.path.getsize(path_e)
        sz = sz // 4
        # print('columns sz {}'.format(sz))   
        columns = np.fromfile(path_e, dtype=np.int32, count=sz)
        columns = torch.from_numpy(columns).to(torch.int64)

        return CSRGraph(columns, row_ptr, directed=False)