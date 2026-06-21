"""
A Graph type implemented with CSRC (CSR + CSC).
"""
from .Graph import Graph
from .CSRGraph import CSRGraph
from .CSCGraph import CSCGraph
import torch
import numpy as np

class CSRCGraph(Graph):
    """
    CSR + CSC implementation of Graph. Efficient access to out_nbrs and in_nbrs. Assume the graph is directed. (otherwise use CSRGraph). Provides a mapping from CSC row indices to CSR column indices.
    """
    def __init__(self,
                 shuffle_ptr: torch.Tensor,
                 columns: torch.Tensor=None,
                 row_ptr: torch.Tensor=None,
                 rows: torch.Tensor=None,
                 column_ptr: torch.Tensor=None,
                 csr: CSRGraph=None,
                 csc: CSCGraph=None,
                 vertex_attrs_list=[],
                 vertex_attrs_tensor: torch.Tensor=None,
                 vertex_attrs_mask: torch.Tensor=None,
                 edge_attrs_list=[],
                 edge_attrs_tensor: torch.Tensor=None,
                 edge_attrs_mask: torch.Tensor=None):
        """
        Initialize a CSRCGraph object with according datatypes (tensors).
        
        :param Tensor columns: out-neighbors of vertex (arranged in order) (for CSR)
        :param Tensor row_ptr: pointers of each vertex for val and col_ind (for CSR)
        :param Tensor rows: in-neighbors of vertex
        (arranged in order) (for CSC)
        :param Tensor column_ptr: pointers of each vertex for val and row_ind (for CSC)
        :param Tensor shuffle_ptr: pointers from CSC rows to CSR columns.  rows = edge_start[shuffle_ptr] (edge_start = rows.sort())
        :param list vertex_attrs_list: list of vertex attributes names
        :param Tensor vertex_attrs_tensor: tensor of vertex attributes that stores data
        :param Tensor vertex_attrs_mask: mask of vertex attributes
        :param list edge_attrs_list: list of edge attributes names
        :param Tensor edge_attrs_tensor: tensor of edge attributes that stores data
        :param Tensor edge_attrs_mask: mask of edge attributes
        """
        # 使用 CSR-C 格式默认为有向图, 否则直接使用CSR格式, 出边即为入边,
        super().__init__(directed=True)
        if csr is not None:
            self.csr = csr
        else:
            self.csr = CSRGraph(columns=columns, row_ptr=row_ptr, directed=True,
                                vertex_attrs_list=vertex_attrs_list, vertex_attrs_tensor=vertex_attrs_tensor, vertex_attrs_mask=vertex_attrs_mask,
                                edge_attrs_list=edge_attrs_list, edge_attrs_tensor=edge_attrs_tensor, edge_attrs_mask=edge_attrs_mask,)
        if csc is not None:
            self.csc = csc
        else:
            self.csc = CSCGraph(rows=rows, column_ptr=column_ptr, directed=True)
        self.shuffle_ptr = shuffle_ptr
                    
    @property
    def num_vertices(self):
        return self.csr.num_vertices
    
    @property
    def num_edges(self):
        return self.csr.num_edges
    
    def out_degree(self, vertices):
        return self.csr.out_degree(vertices)
    
    def in_degree(self, vertices):
        return self.csc.in_degree(vertices)
    
    def out_nbrs(self, vertices):
        return self.csr.out_nbrs(vertices)
    
    def out_nbrs_csr(self, vertices):
        return self.csr.out_nbrs_csr(vertices)
    
    def all_out_nbrs_csr(self):
        return self.csr.all_out_nbrs_csr()
    
    def in_nbrs(self, vertices):
        return self.csc.in_nbrs(vertices)
    
    def in_nbrs_csr(self, vertices):
        return self.csc.in_nbrs_csr(vertices)
    
    def all_in_nbrs_csr(self):
        return self.csc.all_in_nbrs_csr()
    
    def out_edges(self, vertices):
        return self.csr.out_edges(vertices)
    
    def out_edges_csr(self, vertices):
        return self.csr.out_edges_csr(vertices)
    
    def all_out_edges_csr(self):
        return self.csr.all_out_edges_csr()
    
    def in_edges(self, vertices):
        csc_in_edges, csc_masks = self.csc.in_edges(vertices)
        # in_edges = self.shuffle_ptr[csc_in_edges]
        in_edges = torch.where(csc_masks, self.shuffle_ptr[csc_in_edges], torch.ones_like(csc_in_edges) * -1)
        return in_edges, csc_masks
    
    def in_edges_csr(self, vertices):
        csc_in_edges, ptr = self.csc.in_edges_csr(vertices)
        in_edges = self.shuffle_ptr[csc_in_edges]
        return in_edges, ptr
    
    def all_in_edges_csr(self, vertices):
        return self.shuffle_ptr, self.csc.csr.row_ptr
    
    @property
    def device(self):
        return self.csr.device
    
    def to(self, *args, **kwargs):
        if self.vertices_t is not None:
            self.vertices_t = self.vertices_t.to(*args, **kwargs)
        if self.edges_t is not None:
            self.edges_t = self.edges_t.to(*args, **kwargs)
        self.csr.to(*args, **kwargs)
        self.csc.to(*args, **kwargs)
        if self.shuffle_ptr is not None:
            self.shuffle_ptr = self.shuffle_ptr.to(*args, **kwargs)
        
    def pin_memory(self):
        self.csr.pin_memory()
        self.csc.pin_memory()
        if self.shuffle_ptr is not None:
            self.shuffle_ptr = self.shuffle_ptr.pin_memory()
        
    def subgraph(self, vertices: torch.Tensor):
        """
        Get a subgraph induced by the given vertices.
        """
        # First convert to edge list, then convert to CSRCGraph
        edge_starts = []
        edge_ends = []
        for v in vertices:
            for nbr in self.out_nbrs(v):
                if nbr in vertices:
                    edge_starts.append(v)
                    edge_ends.append(nbr)
        return CSRCGraph.edge_list_to_Graph(edge_starts, edge_ends)
    
    def csr_subgraph(self, vertices: torch.Tensor):
        new_csr, indices_csr = self.csr.csr_subgraph(vertices)
        new_csc, indices_csc = self.csc.csr_subgraph(vertices)
        new_shuffle_ptr = self.shuffle_ptr[indices_csc]
        return CSRCGraph(csr=new_csr, csc=new_csc, shuffle_ptr=new_shuffle_ptr), \
            indices_csr
    
    def get_vertex_attr(self, vertices, attr):
        return self.csr.get_vertex_attr(vertices, attr)
    
    def select_vertex_by_attr(self, attr, cond):
        return self.csr.select_vertex_by_attr(attr, cond)
    
    def set_vertex_attr(self, vertices, attr, value, mask):
        return self.csr.set_vertex_attr(vertices, attr, value, mask)
    
    def get_edge_attr(self, edges, attr):
        return self.csr.get_edge_attr(edges, attr)
    
    def select_edge_by_attr(self, attr, cond):
        return self.csr.select_edge_by_attr(attr, cond)
    
    def set_edge_attr(self, edges, attr, value, mask):
        return self.csr.set_edge_attr(edges, attr, value, mask)
        
    @staticmethod
    def edge_list_to_Graph(edge_starts, edge_ends, vertices=None, edge_attrs=None, edge_attrs_list=[], vertex_attrs=None, vertex_attrs_list=[]):
        # get vertex to index mapping  注意此时的Vertex本身有序且为Unique
        vertex_to_index = {}
        vertex_data_list = [[] for _ in range(len(vertex_attrs_list))]
        for vertex, index in zip(vertices, range(len(vertices))):
            if index == 0:
                import logging
                logging.info(vertex)
            vertex_to_index[vertex] = index
            if vertex_attrs is not None:
                for data_index, data in enumerate(vertex_attrs):
                    vertex_data_list[data_index].append(data[vertex])
        # Change edge_starts and edge_ends to indices by dictionary 'vertex_to_index'
        edge_starts = torch.LongTensor([vertex_to_index[i] for i in edge_starts])
        edge_ends = torch.LongTensor([vertex_to_index[i] for i in edge_ends])
        # print("edge_starts: {}".format(edge_starts))
        # print("edge_ends: {}".format(edge_ends))
        data_tensors = [torch.FloatTensor(i) for i in edge_attrs]
        # Conduct counter sort
        row_ptr, pos_sources = CSRCGraph.counter_sort(edge_starts, len(vertices))
        columns = edge_ends[pos_sources]

        # 根据pos_sources 来再次更新 edge_start  edge_end
        # 此时edge_ends = columns
        edge_starts = edge_starts[pos_sources]
        edge_ends = edge_ends[pos_sources]
        for t in data_tensors:
            t = t[pos_sources]
        column_ptr, pos_targets = CSRCGraph.counter_sort(edge_ends, len(vertices))
        rows = edge_starts[pos_targets]
        
        if len(data_tensors) != 0:
            edge_attrs_tensor = torch.stack(data_tensors, dim=0)
            edge_attrs_mask = torch.ones(edge_attrs_tensor.shape, dtype=torch.bool)
        else:
            edge_attrs_tensor, edge_attrs_mask = None, None
        if vertex_attrs is not None:
            vertex_attrs_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in vertex_data_list], dim=0)
            vertex_attrs_mask = torch.ones(vertex_attrs_tensor.shape, dtype=torch.bool)
        else:
            vertex_attrs_tensor = None
            vertex_attrs_mask = None
        return CSRCGraph(
            shuffle_ptr=pos_targets,
            columns=columns,
            row_ptr=row_ptr,
            rows=rows,
            column_ptr=column_ptr,
            vertex_attrs_tensor=vertex_attrs_tensor,
            vertex_attrs_list=vertex_attrs_list,
            vertex_attrs_mask=vertex_attrs_mask,
            edge_attrs_tensor=edge_attrs_tensor,
            edge_attrs_list=edge_attrs_list,
            edge_attrs_mask=edge_attrs_mask
        ), vertex_to_index, data_tensors
    
    @staticmethod
    def read_graph(f, split=None):
        edge_starts, edge_ends, vertices, data = Graph.read_edgelist(f, split)
        return CSRCGraph.edge_list_to_Graph(edge_starts, edge_ends, vertices=vertices, edge_attrs=data)
    
    @staticmethod   
    def read_csrc_graph(f, split=None):
        """
        直接读取CSRC存储格式的文件
        """
        print('-------- {} ------------'.format(f))
        f = open(f, 'r')
        # info = np.loadtxt(f, dtype=np.int32, max_rows=1)
        # print('info read done')
        # row_ptr = np.loadtxt(f, dtype=np.int32, max_rows=1)
        # print('row_ptr read done')
        # columns = np.loadtxt(f, dtype=np.int32, max_rows=1)
        # print('columns read done')
        # col_ptr = np.loadtxt(f, dtype=np.int32, max_rows=1)
        # print('col_ptr read done')
        # rows = np.loadtxt(f, dtype=np.int32, max_rows=1)
        # print('rows read done')
        # vertex_cnt = info[0]
        # edge_cnt = info[1]
        # print(vertex_cnt, " ", edge_cnt)
        import time
        lines = f.readlines()
        info = lines[0].split(' ')
        vertex_cnt = int(info[0])
        edge_cnt = int(info[1])
        print(vertex_cnt, " ", edge_cnt)
        row_ptr = lines[1].split(' ')
        row_ptr = [int(i) for i in row_ptr[:-1]]

        columns = lines[2].split(' ')
        columns = [int(i) for i in columns[:-1]]

        col_ptr = lines[3].split(' ')
        col_ptr = [int(i) for i in col_ptr[:-1]]
        rows = lines[4].split(' ')
        rows = [int(i) for i in rows[:-1]]
        print(len(row_ptr), " ", len(columns), " ", len(col_ptr), " ", len(rows))
        shuffle_ptr = None
        # # generate csrc graph
        # print(type(row_ptr[0]))
        # print(len(row_ptr), " ", len(columns), " ", len(col_ptr), " ", len(rows))
        # i = 0
        # info = None
        # row_ptr = None
        # columns = None
        # col_ptr = None
        # rows = None
        # content = [None, None, None, None, None]

        # for line in f:
        #     content[i] = line.split(' ')
        #     print(len(content[i]))
        #     if i > 0:
        #         content[i] = content[i][:-1]
        #     for j in range(len(content[i])):
        #         content[i][j] = int(content[i][j])
        #     i += 1
        # info = content[0]
        # vertex_cnt = info[0]
        # edge_cnt = info[1]
        # print(vertex_cnt, " ", edge_cnt)
        # row_ptr = content[1]
        # columns = content[2]
        # col_ptr = content[3]
        # rows = content[4]
            
        
        return CSRCGraph(
            shuffle_ptr=None,
            columns=torch.tensor(columns, dtype=torch.long),
            row_ptr=torch.tensor(row_ptr, dtype=torch.long),
            rows=torch.tensor(rows, dtype=torch.long),
            column_ptr=torch.tensor(col_ptr, dtype=torch.long),
        )
    @staticmethod
    def read_csrc_graph_pandas(f, split=None):
        if split == None:
            split = ' '
        print('-------- {} ------------'.format(f))
        import pandas as pd

        # pandas 读取数据
        data = pd.read_csv(f, sep=split, header=None, nrows=1)
        info = data.values[0]
        vertex_cnt = info[0]
        edge_cnt = info[1]
        print('vertex_cnt {}  edge_cnt {}'.format(vertex_cnt, edge_cnt))
        print('memory: {}'.format(data.info()))

        # pandas 读取数据
        data = pd.read_csv(f, sep=split, header=None, nrows=1, skiprows=1, dtype=np.int32)
        columns = data.values[0]
        del data
        columns = torch.from_numpy(columns).long()
        print(1)

        data = pd.read_csv(f, sep=split, header=None, nrows=1, skiprows=2, dtype=np.int32)
        row_ptr = data[0]
        del data
        row_ptr = torch.from_numpy(row_ptr).long()
        print(2)

        data = pd.read_csv(f, sep=split, header=None, nrows=1, skiprows=3, dtype=np.int32)
        rows = data[0]
        del data
        rows = torch.from_numpy(rows).long()
        print(3)

        data = pd.read_csv(f, sep=split, header=None, nrows=1, skiprows=4, dtype=np.int32)
        col_ptr = data[0]
        del data
        col_ptr = torch.from_numpy(col_ptr).long()
        print(4)
        
        return CSRCGraph(
            shuffle_ptr=None,
            columns=columns,
            row_ptr=row_ptr,
            rows=rows,
            column_ptr=col_ptr
        )

    @staticmethod
    def read_csrc_graph_bin(f, split=None):
        """
        read csrc graph from binary file.

        兼容两种文件命名约定：
          - 旧版：`csr_vlist` / `csr_elist` / `csc_vlist` / `csc_elist`（无后缀）
          - 新版：`csr_vlist.bin` / `csr_elist.bin`（CSC 缺失时自动从 CSR 构造）

        若 CSC 文件不存在，会基于 CSR 现场构造（O(E) 时间），无需外部预处理。
        """
        print('reading csrc graph from binary file: {}'.format(f))
        import os

        def _resolve_path(base, suffix):
            """优先尝试 .bin 后缀，回退到无后缀。"""
            for cand in (base + suffix + '.bin', base + suffix):
                if os.path.exists(cand):
                    return cand
            return None

        path0 = _resolve_path(f, '/csr_vlist')
        path1 = _resolve_path(f, '/csr_elist')
        path2 = _resolve_path(f, '/csc_vlist')
        path3 = _resolve_path(f, '/csc_elist')

        if path0 is None or path1 is None:
            raise FileNotFoundError(
                f"CSR files not found in {f}/ (expected csr_vlist[.bin] + csr_elist[.bin])")

        x = 4  # 4(int) or 8(long)
        # 读取 csr_vlist
        sz = os.path.getsize(path0) // x
        row_ptr = np.fromfile(path0, dtype=np.int32, count=sz)
        row_ptr = torch.from_numpy(row_ptr).long()
        # 读取 csr_elist
        sz = os.path.getsize(path1) // 4
        columns = np.fromfile(path1, dtype=np.int32, count=sz)
        columns = torch.from_numpy(columns).long()

        # CSC 文件缺失时从 CSR 自动构造
        if path2 is None or path3 is None:
            print('[CSRCGraph] CSC files not found, constructing CSC from CSR on-the-fly...')
            num_verts = row_ptr.shape[0] - 1
            # 用 bincount + cumsum 构造 CSC：对每条边 (row, col)，统计每 col 的入度
            csr_rows = torch.arange(num_verts, device=row_ptr.device).repeat_interleave(
                row_ptr[1:] - row_ptr[:-1])
            col_ptr = torch.cumsum(
                torch.bincount(columns, minlength=num_verts), dim=-1)
            col_ptr = torch.cat((torch.tensor([0], dtype=col_ptr.dtype, device=col_ptr.device),
                                 col_ptr))
            # 按 column 排序得到 CSC 的 rows（稳定排序保证相同 column 内顺序一致）
            order = torch.argsort(columns, stable=True)
            rows = csr_rows[order]  # CSC 的 rows

            # 构造 shuffle_ptr：CSC 第 i 条边对应 CSR 第 shuffle_ptr[i] 条边
            # CSC 边 i = (rows[i], dst_i)，其中 dst_i = searchsorted(col_ptr, i, right=True) - 1
            # CSR 边 j = (csr_rows[j], columns[j])
            # 用 (src, dst) pair 匹配：构造 CSR 边 idx → (src*V + dst, j) 与 CSC 边 idx → (src*V + dst, i)
            # 排序后对应位置匹配
            n_edges = columns.shape[0]
            # CSC 边的 (src*V + dst, csc_idx)
            csc_dst = torch.searchsorted(col_ptr, torch.arange(n_edges, device=row_ptr.device),
                                         right=True) - 1
            csc_keys = rows * num_verts + csc_dst
            csc_order = torch.argsort(csc_keys, stable=True)
            # CSR 边的 (src*V + dst, csr_idx)
            csr_keys = csr_rows * num_verts + columns
            csr_order = torch.argsort(csr_keys, stable=True)
            # shuffle_ptr[csc_idx] = csr_idx for matching edge
            shuffle_ptr = torch.zeros(n_edges, dtype=torch.long, device=row_ptr.device)
            shuffle_ptr[csc_order] = csr_order
        else:
            sz = os.path.getsize(path2) // x
            col_ptr = np.fromfile(path2, dtype=np.int32, count=sz)
            col_ptr = torch.from_numpy(col_ptr).long()
            sz = os.path.getsize(path3) // 4
            rows = np.fromfile(path3, dtype=np.int32, count=sz)
            rows = torch.from_numpy(rows).long()
            # 外部提供的 CSC 文件暂时不构造 shuffle_ptr（保留 None，需用户自行提供或重新计算）
            shuffle_ptr = None

        return CSRCGraph(
            shuffle_ptr=shuffle_ptr,
            columns=columns,
            row_ptr=row_ptr,
            rows=rows,
            column_ptr=col_ptr
        )
    @staticmethod
    def counter_sort(tensor: torch.Tensor, num_vertices):
        """
        Implements counter sort. counts[i] is the number of elements in tensor that are less than or equal to i. pos[i] is the position of the i-th smallest element in tensor.
        """
        counts = torch.cumsum(torch.bincount(tensor, minlength=num_vertices), dim=-1)
        counts = torch.cat((torch.tensor([0]), counts))
        # 获取索引值数组
        """
        以 tensor = edge_start 来举例说明 torch.argsort() 在这里的作用:
        在这里我们需要获取到 columns, 我们知道我们需要的 columns 可以通过 edge_end[pos] 这种映射来获取, 映射数组 pos 即为我们所求
        >>> edge_start = torch.LongTensor([0, 0, 0, 2, 1, 1])
        >>> edge_end   = torch.LongTensor([1, 2, 3, 3, 2, 3])
        >>> pos = torch.argsort(edge_start)
            pos = tensor([0, 1, 2, 4, 5, 3])
        此时pos中的值为edge_start 数组的索引, 按照索引在edge_start数组中对应值的大小从小到大排列
        而 columns 中的值也是按照 row 值大小的顺序进行排列, 和 edge_start相对应
        columns = edge_end[pos]
        """
        pos = torch.argsort(tensor)
        return counts, pos
