"""
Helper functions used in the TCRGraph framework.
"""
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch_scatter
import heapq
from torch_scatter import segment_csr
import time

# from torch_geometric
def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)

# from torch_geometric
def scatter(src: Tensor, index: Tensor, dim: int = 0,
            dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:
    if index.dim() != 1:
        raise ValueError(f"The `index` argument must be one-dimensional "
                            f"(got {index.dim()} dimensions)")

    dim = src.dim() + dim if dim < 0 else dim

    if dim < 0 or dim >= src.dim():
        raise ValueError(f"The `dim` argument must lay between 0 and "
                            f"{src.dim() - 1} (got {dim})")

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    size = list(src.size())
    size[dim] = dim_size

    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == 'mean':
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    if reduce == 'min' or reduce == 'max' or reduce == 'mul':
        return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                        reduce=reduce)
    raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")

def csr_to_dense(src: Tensor, indptr: Optional[Tensor] = None,
                 fill_value: float = 0., max_degree: Optional[int] = None,
                 batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    Convert a CSR list to a neighbor list. Based on PyG to_dense_batch.
    """
    # indptr -> batch
    if indptr is None:
        batch = None
    else:
        ranges = torch.arange(indptr.size(0) - 1, device=indptr.device)
        diffs = torch.diff(indptr)
        batch = torch.repeat_interleave(ranges, diffs)
    
    if batch is None and max_degree is None:
        mask = torch.ones(1, src.size(0), dtype=torch.bool, device=src.device)
        return src.unsqueeze(0), mask

    if batch is None:
        batch = src.new_zeros(src.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(src.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    filter_nodes = False
    if max_degree is None:
        max_degree = int(num_nodes.max())
    elif num_nodes.max() > max_degree:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=src.device) - cum_nodes[batch]
    idx = tmp + (batch * max_degree)
    if filter_nodes:
        mask = tmp < max_degree
        src, idx = src[mask], idx[mask]

    size = [batch_size * max_degree] + list(src.size())[1:]
    out = src.new_full(size, fill_value)
    out[idx] = src
    out = out.view([batch_size, max_degree] + list(src.size())[1:])

    mask = torch.zeros(batch_size * max_degree, dtype=torch.bool,
                       device=src.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_degree)

    return out, mask

def batched_csr_selection(starts, ends):
    """
    Given start indices and end indices, give its CSR selection tensor and pointer.
    Example:
    starts [0, 2, 5, 18]
    ends   [2, 5, 9, 21]
    
    returns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20]
    ptr: [0, 2, 5, 9, 12]
    """
    device = starts.device
    sizes = ends - starts
    begin_idx = sizes.cumsum(0)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), begin_idx])
    # begin_idx = ptr[:-1]
    # begin_idx = begin_idx.roll(1)
    # begin_idx[0] = 0
    begin_idx = ptr[:-1]
    result = torch.arange(ptr[-1], device=device) - (begin_idx - starts).repeat_interleave(sizes)
    # print("batched_csr_selection sizes {} ".format(sizes.sum()))
    return result, ptr

def batched_adj_selection(starts, ends, mask_value=-1):
    """
    For example:
    starts [0, 2, 5, 18]
    ends   [2, 5, 9, 21]
    
    returns (assuming mask_value=-1)
    [[ 0,  1, -1, -1],
     [ 2,  3,  4, -1],
     [ 5,  6,  7,  8],
     [18, 19, 20, -1]],
    and the according mask
    """
    device = starts.device
    sizes = ends - starts
    begin_idx = sizes.cumsum(0)
    max_size = torch.max(sizes)
    result = torch.ones((starts.size(0) * max_size,), dtype=torch.int64, device=device) * mask_value
    begin_idx = begin_idx.roll(1)
    begin_idx[0] = 0
    ranges = torch.arange(sizes.sum(), device=device)
    value = ranges - (begin_idx - starts).repeat_interleave(sizes)
    row_starts = torch.arange(starts.size(0), device=device) * max_size
    idx = ranges + (row_starts - begin_idx).repeat_interleave(sizes)
    result[idx] = value
    result = result.view((starts.size(0), max_size))
    mask = (result != mask_value)
    return result, mask

def divide_equally(data: torch.Tensor, partition_size):
    """
    Partition data into `partition_size` groups, and keep the sum of each group as near as possible.
    Return a list of tensors, which includes the indices of the partitioned data.

    """
    t1 = time.time()
    # data[indices] = sorted
    # data[indices[i]] = sorted[i]
    # indices is vertex id
    sorted, indices = torch.sort(data, descending=True)
    partition_size = min(partition_size, data.size(0))
    heap = [(0, idx) for idx in range(partition_size)]
    heapq.heapify(heap)
    print('heap', heap)
    results = [[] for _ in range(partition_size)]
    value_results = [[] for _ in range(partition_size)]
    data_idx = 0
    while data_idx < data.size(0):
        set_sum, idx = heapq.heappop(heap)
        results[idx].append(indices[data_idx])
        value_results[idx].append(sorted[data_idx])
        set_sum += sorted[data_idx]
        heapq.heappush(heap, (set_sum, idx))
        data_idx += 1
    t2 = time.time()
    print(f"divide_equally: {t2 - t1}")
    return [torch.tensor(result, device=data.device, dtype=torch.long) for result in results], \
        [torch.tensor(result, device=data.device) for result in value_results]

# generate SubGraph from Graph
# def get_partition(graph: Graph, partition_num: int):
#     row_ptr = None
#     columns = None
#     degrees = None

#     if isinstance(graph, CSRCGraph):
#         degrees = graph.csc.csr.out_degrees
#         row_ptr = graph.csc.csr.row_ptr
#         columns = graph.csc.csr.columns
#     elif isinstance(graph, CSRGraph):
#         degrees = graph.out_degrees
#         row_ptr = graph.row_ptr
#         columns = graph.columns

    
#     # print(degrees.dtype)
#     row_ptr = row_ptr.to(torch.long)
#     degrees = degrees.to(torch.long)
#     degrees_ptr = row_ptr[1:] - row_ptr[:-1]
#     mask = degrees_ptr == degrees
#     # if mask.sum() != len(degrees):
#     #     print('not equal')
#     # else:
#     #     print('equal')
#     # print('row_ptr size {}'.format(row_ptr[-1]))
#     # print('sum_degrees {}'.format(degrees.sum()))
#     # print('max_degrees {}'.format(torch.max(degrees)))
#     # print('part_sum {}'.format(torch.sum(degrees[0:30000000])))
#     # print('part_sum {}'.format(torch.sum(degrees[30000000:41652231])))
#     degrees_sum = torch.cumsum(degrees, dim=0, dtype=torch.int64)
#     all_sum = degrees_sum[-1]


#     # print('all_sum {}'.format(all_sum))
#     partition_size = all_sum // partition_num
#     target_sizes = [i * partition_size for i in range(1, partition_num)]
#     target_sizes = torch.tensor(target_sizes, dtype=torch.int64)
#     indexs = torch.searchsorted(degrees_sum, target_sizes)

#     vertex_begin_idx = [0]
#     for i in range(partition_num - 1):
#         vertex_begin_idx.append(indexs[i] + 1)
#     vertex_begin_idx.append(graph.num_vertices)
#     vertex_begin_idx = torch.tensor(vertex_begin_idx, dtype=torch.int64)

#     parts_row_ptr = []
#     parts_columns = []

#     for i in range(partition_num):
#         part_degrees = degrees[vertex_begin_idx[i]:vertex_begin_idx[i + 1]]
#         part_row_ptr = torch.cumsum(part_degrees, dim=0)
#         part_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int64), part_row_ptr])
#         # part_row_ptr = graph.csc.csr.row_ptr[vertex_begin_idx[i]:vertex_begin_idx[i + 1]].clone()
#         # print('partition {} begin {} end {}'.format(i, vertex_begin_idx[i], vertex_begin_idx[i + 1]))
#         col_begin = row_ptr[vertex_begin_idx[i]]
#         col_end = row_ptr[vertex_begin_idx[i + 1]]
#         part_columns = columns[col_begin : col_end]
        
#         # print('partiton {} row_ptr_sum {} columns_sum{}'.format(i, len(part_row_ptr), len(part_columns)))
#         # 从 0 开始
#         # part_graph = PartitionGraph(part_row_ptr, part_columns, vertex_begin_idx, i)
#         part_row_ptr = part_row_ptr.to(torch.long)
#         part_columns = part_columns.to(torch.long)
#         parts_row_ptr.append(part_row_ptr)
#         parts_columns.append(part_columns)
    
#     return parts_row_ptr, parts_columns, vertex_begin_idx

if __name__ == '__main__':
    src = torch.tensor([0,1,2,3,4,5])
    indptr = torch.tensor([0,2,3,6])
    print(csr_to_dense(src, indptr))
    