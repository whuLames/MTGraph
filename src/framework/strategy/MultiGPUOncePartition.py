"""
In this strategy, partition is only done once. The partitioned subgrpahs will be managed by their respective processes.
"""
import sys
from . import Strategy
from ..partition import Partition
import torch
import torch.multiprocessing as mp
from src.type.Subgraph import Subgraph
import queue
from copy import deepcopy
import threading
import select
import time

# one process to compute subgraph on GPUs
class ComputeOnOneGPUProcess(mp.Process):
    def __init__(self, process_num, activated_vertices_mask, changed_vertex_data_mask,
                 changed_edge_data_mask, subgraph, prog, message_pipe, process_pipes_1, process_pipes_2, rank, vertex_to_process_id, **kwargs):
        super().__init__(**kwargs)
        self.subgraph = subgraph
        self.prog = prog
        self.message_pipe = message_pipe    # to transfer data with parent process
        self.process_pipes_1 = process_pipes_1  # a list of pipes to transfer data with other processes (sending end)
        self.process_pipes_2 = process_pipes_2 # a list of pipes to transfer data with other processes (receiving end)
        self.rank = rank
        self.activated_vertices_mask = activated_vertices_mask
        self.changed_vertex_data_mask = changed_vertex_data_mask
        self.changed_edge_data_mask = changed_edge_data_mask
        self.vertex_to_process_id = vertex_to_process_id
        # number of processes
        self.process_num = process_num
        
    def run(self):
        self.prog.to(f'cuda:{self.rank}', non_blocking=True)
        self.subgraph.to(f'cuda:{self.rank}', non_blocking=True)
        prog = self.prog
        subgraph = self.subgraph
        # activated_vertices_mask = self.activated_vertices_mask
        activated_vertices_mask = self.activated_vertices_mask.to(f'cuda:{self.rank}', non_blocking=True)
        # do computation
        vertices = subgraph.sub_vertices  # origin graph vertices
        indices = prog.scatter_nbrs(vertices)
        prog.set_graph(subgraph)  
        
        # store: who to send, what to send
        changed_vertices = [[] for _ in range(self.process_num)]
        changed_vertices_data = [[] for _ in range(self.process_num)]
        
        # send message that I'm ready
        self.message_pipe.send(self.rank)
            
        while True:
            # send changed vertex data to other processes
            ta = time.time()
            for p in range(self.process_num):
                if p != self.rank:
                    pipe = self.process_pipes_1[p]
                    if pipe.closed:
                        continue
                    vertices_send = changed_vertices[p]
                    vertices_send_data = changed_vertices_data[p]
                    pipe.send((vertices_send, vertices_send_data))
                changed_vertices[p] = []
                changed_vertices_data[p] = []
            # receive data from other processes
            for p in range(self.process_num):
                if p != self.rank:
                    pipe = self.process_pipes_2[p]
                    if pipe.closed:
                        continue
                    vertices_recv, vertices_recv_data = pipe.recv()
                    if len(vertices_recv) != 0:
                        self.prog.vertex_data[vertices_recv] = vertices_recv_data.to(f'cuda:{self.rank}', non_blocking=True)
            tb = time.time()
            print(f'rank {self.rank} send and receive time: {tb - ta}s')

            # compute
            ta = time.time()
            g_nbrs, g_edges, g_ptr = prog.gather_nbrs(vertices)
            gd, gd_ptr = prog.gather(vertices, g_nbrs, g_edges, g_ptr)
            gsum = prog.sum(gd, gd_ptr)
            apply_d, apply_mask = prog.apply(vertices, gsum)
            prog.vertex_data[vertices] = torch.where(apply_mask,
                apply_d, prog.vertex_data[vertices])
            s_nbrs, s_edges, s_ptr = prog.scatter_nbrs(vertices)
            s_data, s_mask = prog.scatter(vertices, s_nbrs, s_edges, s_ptr, apply_d)
            if s_data is not None and s_mask is not None:
                prog.edge_data[indices] = torch.where(s_mask, s_data, prog.edge_data[indices])
            # TODO: add edge data changes
            # self.changed_edge_data_mask[indices] = self.rank
            # self.changed_vertex_data_mask[vertices] = self.rank
            tb = time.time()
            print(f'rank {self.rank} compute time for one iter: {tb - ta}s')
            # update activated vertices
            not_change_activated = prog.not_change_activated
            if not not_change_activated:
                # activated_vertices_mask[:] = torch.logical_or(activated_vertices_mask, prog.next_activated_vertex_mask.to('cuda:0', non_blocking=True))
                # print('rank', self.rank)
                # print('mask', activated_vertices_mask)
                activated_vertices_mask[:] = torch.logical_or(activated_vertices_mask, prog.next_activated_vertex_mask.to(f'cuda:{self.rank}', non_blocking=True))
            # sort changed vertices
            # apply_mask is not necessarily one-dimensional
            ta = time.time()
            # 当前更新的节点就是 apply_mask 为 True 的节点
            current_vertex_changed = vertices[apply_mask[:, 0]]
            current_vertex_changed_data = prog.vertex_data[current_vertex_changed]
            for v, d in zip(current_vertex_changed, current_vertex_changed_data):
                processes = self.vertex_to_process_id[v.item()]
                for p in processes:
                    if p != self.rank:
                        changed_vertices[p].append(v.cpu())
                        changed_vertices_data[p].append(d.cpu())
            for p in processes:
                if p != self.rank:
                    if len(changed_vertices) > 0:
                        changed_vertices[p] = torch.stack(changed_vertices[p])
                        changed_vertices_data[p] = torch.stack(changed_vertices_data[p])
                
            # notify the parent process that everything is done in this iteration
            self.message_pipe.send(self.rank)
            # wait for the reduction on the parent sides
            self.message_pipe.recv()
            tb = time.time()
            print(f'rank {self.rank} prepare data time: {tb - ta}s')
            if not prog.not_change_activated or (prog.nbr_update_freq != 0 and prog.curr_iter % prog.nbr_update_freq == 0): 
                print('sub_vertices ', subgraph.sub_vertices.device)
                print('activated_vertices_mask ', activated_vertices_mask.device)
                print('x', activated_vertices_mask[subgraph.sub_vertices].device)
                vertices = subgraph.sub_vertices[activated_vertices_mask[subgraph.sub_vertices]]
                if vertices.numel() == 0:  # 此时无激活节点, 代表计算结束
                    # notify the parent process that I'm done
                    self.message_pipe.send(-1 - self.rank)
                    # close all pipes
                    self.process_pipes_1[self.rank].close()
                    self.process_pipes_2[self.rank].close()
                    break
                # indices = prog.scatter_nbrs(vertices)
                
            prog.not_change_activated = False
            prog.curr_iter += 1

class MultiGPUOncePartition(Strategy.Strategy):
    def __init__(self, partition: Partition, program, devices=1) -> None:
        super().__init__()
        self.partition = partition
        self.devices = devices
        self.program = program
        self.sub_in_degrees, self.sub_out_degrees = None, None
        
        self.activated_vertices_mask = torch.zeros(self.program.graph.vertices.numel(), dtype=torch.bool, device='cuda')
        
        self.changed_vertex_data_mask = torch.ones_like(self.program.vertex_data, 
                                                        dtype=torch.int16, device=torch.device('cuda')) * -1
        self.changed_edge_data_mask = torch.ones_like(self.program.edge_data, 
                                                      dtype=torch.int16, device=torch.device('cuda')) * -1
            
    def compute(self):
        ta = time.time()
        prog = self.program
        graph = prog.graph
        mp.set_start_method('spawn', force=True)

        # first partition graphs, each for a process
        self.partition.set_num_partitions(self.devices)
        self.partition.set_graph(graph)
        partitions = self.partition.generate_partitions()
        subgraphs = []
        # keep record of vertex -> Processes related to the vertex. Stored in CPU
        # related == in that subgraph
        vertex_to_process_id = {v.item(): set() for v in graph.vertices}
        print(f'Partitioned into {len(partitions)} subgraphs')
        for num, ((p, i), v) in enumerate(partitions):
            # 这里应该是 origin_to_sub
            sub_to_orig_vertices = torch.zeros_like(prog.graph.vertices)
            sub_to_orig_vertices[v] = torch.arange(v.numel())
            # sub_to_orig_edges = torch.zeros_like(prog.graph.edges)
            # sub_to_orig_edges[i] = torch.arange(i.numel())
            new_subgraph = Subgraph(p, v, i, sub_to_orig_vertices)
            subgraphs.append(new_subgraph)
            
            # collect vertex -> process id
            for vertex in v:
                vertex_to_process_id[vertex.item()].add(num)
            for v in new_subgraph.all_out_nbrs_csr()[0]:
                vertex_to_process_id[v.item()].add(num)
            for v in new_subgraph.all_in_nbrs_csr()[0]:
                vertex_to_process_id[v.item()].add(num)
            
            print(f"Subgraph {num} has {v.numel()} vertices and {i.numel()} edges")
            
        # create processes
        processes = []
        pipes = []
        progs = []
        
        process_pipes_1 = []
        process_pipes_2 = []
        for device in range(self.devices):
            conn1, conn2 = mp.Pipe()
            process_pipes_1.append(conn1)
            process_pipes_2.append(conn2)
        
        for device in range(self.devices):
            conn1, conn2 = mp.Pipe() # conn1 conn2 对应的一对管道
            new_prog = deepcopy(prog)
            new_prog.to(torch.device('cuda', device))
            process = ComputeOnOneGPUProcess(self.devices, self.activated_vertices_mask, self.changed_vertex_data_mask, self.changed_edge_data_mask,
                                             subgraphs[device], new_prog, conn1,
                                             process_pipes_1, process_pipes_2, device, vertex_to_process_id)
            processes.append(process)
            pipes.append(conn2)
            progs.append(new_prog)
            
        for p in processes:
            p.start()
        
        # wait for all processes to be ready
        for p in pipes:
            p.recv()
        print('All processes ready.')
        
        t1 = None
        # start computation
        ready_processes = set()
        exited_processes = set()
        while True:
            ta = time.time()
            readable, _, _ = select.select(pipes, [], [])
            for r in readable:
                rank = r.recv()
                if rank >= 0:
                    ready_processes.add(rank)
                else:
                    exited_processes.add(rank)
                    # pipes[-rank - 1].send(None)
            # exit if all processes have exited
            if len(exited_processes) == self.devices:
                break
            # are all processes ready?
            if len(ready_processes) + len(exited_processes) == self.devices:
                ready_processes.clear()
                prog.curr_iter += 1
                prog.not_change_activated = False

                if t1 is None:
                    t1 = time.time()
                print(f'Iteration {prog.curr_iter} completed.')
                # Notify that the reduction is completed
                for i in range(self.devices):
                    pipes[i].send(None)
            tb = time.time()
            print(f'time for one iter : {tb - ta}s')
            
        t2 = time.time()
        print(f"Time for calculation: {t2 - t1}s")
        
        vertex_data = prog.vertex_data
        edge_data = prog.edge_data
        return vertex_data, edge_data
        