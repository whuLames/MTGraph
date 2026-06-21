"""
A more complex strategy that runs computation in one process and run subgraph fetching + partitioning in another process.
"""
import sys
from . import Strategy
from ..partition import Partition
import torch
from src.type.Subgraph import Subgraph
import queue
from copy import deepcopy
import threading
import time
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

class FetchPartitionsThread(threading.Thread):
    def __init__(self, activated_vertices_mask, lock, partition_queue, program, strategy, **kwargs):
        super().__init__(**kwargs)
        self.lock = lock
        self.partition_queue = partition_queue
        self.program = program
        self.sub_in_degrees, self.sub_out_degrees = None, None
        self.partition = strategy.partition
        self.strategy = strategy
        self.count = 0
    # run in a separate process; fetches partitions of graphs and put into queue
    def run(self):
        partition_queue = self.partition_queue
        prog = self.program
        if self.sub_out_degrees is None and self.sub_in_degrees is None:
            sub_out_degrees, sub_in_degrees = None, None
            try:
                sub_out_degrees = prog.graph.out_degree(prog.graph.vertices)
            except:
                ...
            try:
                sub_in_degrees = prog.graph.in_degree(prog.graph.vertices)
            except:
                ...
            self.sub_out_degrees, self.sub_in_degrees = sub_out_degrees, sub_in_degrees
        else:
            sub_out_degrees, sub_in_degrees = self.sub_out_degrees, self.sub_in_degrees
        
        while True:
            with self.lock: # 应该把切割过程也放入lock中, 避免出现还没切割完成时，move线程把上一次的切割结果放入GPU队列
                while not self.strategy.changed_mask: #changed_mask == true时才会进行更新
                    self.lock.wait()
                self.strategy.changed_mask = False
                activated_vertices = torch.nonzero(prog.activated_vertex_mask).squeeze()
                self.count = self.count + 1
                logging.info('Partition count : {} num: {}'.format(self.count, activated_vertices))
            if activated_vertices is None or len(activated_vertices.shape) == 0 or activated_vertices.shape[0] == 0:
                partition_queue.put(None)
                print(1)
                break
            subgraph, indices = prog.graph.csr_subgraph(activated_vertices)
            self.partition.set_graph(subgraph)
            partitions = self.partition.generate_partitions()
            new_subgraphs = []
            for (p, i), v in partitions: # p is the subgraph(CSRCGraph) , i is the indices and v is the vertices of subgraph
                new_indices = indices[i] # origin_graph.columns[new_indices] = p.columns
                new_vertices = activated_vertices[v] # new_vertices is subset of origin vertex set
                orig_to_sub_vertices = torch.zeros_like(prog.graph.vertices, device=subgraph.device)
                orig_to_sub_vertices[new_vertices] = torch.arange(new_vertices.numel(), device=subgraph.device)
                # sub_to_orig_edges = torch.zeros_like(prog.graph.edges, device=subgraph.device)
                # sub_to_orig_edges[new_indices] = torch.arange(new_indices.numel(), device=subgraph.device)
                new_subgraph = Subgraph(p, new_vertices, new_indices, orig_to_sub_vertices)
                new_subgraphs.append(new_subgraph)
            partition_queue.put(new_subgraphs)
            del activated_vertices

# run with multithreading (same process with compute_on_gpu); moves partitions to GPU
class MoveToGPUThread(threading.Thread):
    def __init__(self, partition_queue, gpu_queue, **kwargs):
        super().__init__(**kwargs)
        self.partition_queue = partition_queue
        self.gpu_queue = gpu_queue
        self.count = 0
    def run(self):
        last_subgraphs = []
        while True:
            got_from_partition_queue = True
            subgraphs = None
            try:
                subgraphs = self.partition_queue.get_nowait()
            except queue.Empty:
                got_from_partition_queue = False

            if got_from_partition_queue:
                if subgraphs is None:
                    self.gpu_queue.put(None)
                    break
                last_subgraphs = subgraphs
            else:
                if len(last_subgraphs) == 0:  # 仍未获取subgraph，继续获取
                    continue
            for sub in last_subgraphs:
                # sub = deepcopy(sub)
                # assert sub.device == torch.device('cpu')
                sub.to('cuda', non_blocking=True)
                self.gpu_queue.put(sub)

            self.gpu_queue.put(None)  # None is the end flag, indicating the end of a iteration
            # time.sleep(1)
            self.count += 1
            # print('has completed {}'.format(self.count))
        last_subgraphs = []
        self.gpu_queue.put(None)
        
# multithreaded (same process with move_to_gpu); run computations on GPU
class ComputeOnGPUThread(threading.Thread):
    def __init__(self, lock, gpu_queue, prog, strategy, **kwargs):
        super().__init__(**kwargs)
        self.lock = lock
        self.gpu_queue = gpu_queue
        self.prog = prog
        self.vertex_data = None
        self.edge_data = None
        self.strategy = strategy
        self.count = 0
    def run(self):
        last_none = False # the mean?
        prog = self.prog
        prog.to('cuda')
        nbr_update_freq = prog.nbr_update_freq
        activated_vertices_mask = torch.zeros(prog.graph.vertices.numel(), dtype=torch.bool, device='cuda')
        # activated_vertices_mask = torch.zeros(prog.graph.vertices.numel(), dtype=torch.bool)
        
        t1 = None
        count = 0
        while True:
            # get a partitioned graph from queue
            # if queue is empty the thread will wait until the queue is not empty
            part = self.gpu_queue.get()
            if t1 is None:
                # print('First subgraph ready')
                t1 = time.time()
            if part is None:
                if last_none:
                    break
                if not prog.not_change_activated or (nbr_update_freq != 0 and prog.curr_iter % nbr_update_freq == 0): 
                    with self.lock:
                        self.lock.notify()
                        prog.activated_vertex_mask = prog.next_activated_vertex_mask
                        self.strategy.changed_mask = True
                        self.count = self.count + 1
                        # print('Com Count {} iters: {} act: {}'.format(self.count, prog.curr_iter, prog.activated_vertex_mask))
                    prog.next_activated_vertex_mask = torch.zeros_like(prog.activated_vertex_mask, dtype=torch.bool)
                    # activated_vertices_mask = torch.zeros_like(activated_vertices_mask, dtype=torch.bool, device='cuda')
                    activated_vertices_mask = torch.zeros_like(activated_vertices_mask, dtype=torch.bool) # comment out is OK
                    
                prog.curr_iter += 1
                prog.not_change_activated = False
                last_none = True
                continue
            logging.info('迭代轮次 {}'.format(prog.curr_iter))
            last_none = False
            part.to('cuda')
            subgraph = part
            vertices = subgraph.sub_vertices
            indices = subgraph.sub_edges
            prog.set_graph(subgraph)

            g_nbrs, g_edges, g_ptr = prog.gather_nbrs(vertices)
            gd, gd_ptr = prog.gather(vertices, g_nbrs, g_edges, g_ptr)
            gsum = prog.sum(gd, gd_ptr)
            apply_d, apply_mask = prog.apply(vertices, gsum)
            if apply_d is not None and apply_mask is not None:
                # 在这里做全局节点数据的更新, 也就是说节点数据一直维护在GPU之上，我们只需要在不断将各个子图的邻接信息放到GPU即可
                # 因为对于大部分图，我的节点长度是可以接受的
                prog.vertex_data[vertices] = torch.where(apply_mask,
                    apply_d, prog.vertex_data[vertices])
            s_nbrs, s_edges, s_ptr = prog.scatter_nbrs(vertices)
            s_data, s_mask = prog.scatter(vertices, s_nbrs, s_edges, s_ptr, apply_d)
            if s_data is not None and s_mask is not None:
                prog.edge_data[indices] = torch.where(s_mask, s_data, prog.edge_data[indices])
                
            # update activated vertices
            # put new activated vertices into queue
            # if nbr_update_freq = 0, not_change_activated if always True
            # 这个地方感觉多余
            if not prog.not_change_activated:
                activated_vertices_mask = torch.logical_or(activated_vertices_mask, prog.next_activated_vertex_mask)
                
            # del part
            part.to('cpu', non_blocking=True)
            count += 1
            # print('count: ', count)
            pass

        t2 = time.time()
        all_time = t2 - t1
        print('all time: ', all_time)
        prog.to('cpu')
        self.vertex_data = prog.vertex_data
        self.edge_data = prog.edge_data
        
class PartitionStrategy(Strategy.Strategy):
    def __init__(self, program, partition: Partition, max_subgraphs_in_gpu=1) -> None:
        super().__init__()
        self.program = program
        self.partition = partition
        self.max_subgraphs_in_gpu = max_subgraphs_in_gpu
        # Two separtate streams for data transfer and computation        
        self.sub_out_degrees = None
        self.sub_in_degrees = None
        self.changed_lock = threading.Condition() # lock: whether the activated_vertices has changed
        self.changed_mask = True
            
    def compute(self):
        prog = self.program
        # mp.set_start_method('spawn', force=True)
        # partitioned graph (still in CPU)
        partition_queue = queue.Queue()
        # partitioned graph transferred to GPU
        gpu_queue = queue.Queue(self.max_subgraphs_in_gpu)
        # result queue: p3 -> main process 
        p1 = FetchPartitionsThread(prog.activated_vertex_mask, self.changed_lock, partition_queue, self.program, self)
        p2 = MoveToGPUThread(partition_queue, gpu_queue)
        p3 = ComputeOnGPUThread(self.changed_lock, gpu_queue, prog, self)
        with self.changed_lock:
            self.changed_lock.notify()
        p1.start()
        p2.start()
        p3.start()
        
        p1.join()
        p2.join()
        p3.join()
        vertex_data, edge_data = p3.vertex_data, p3.edge_data
        
        return vertex_data, edge_data
        