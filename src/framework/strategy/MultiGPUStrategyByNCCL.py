"""
Multi GPU Compute Strategy by NCCL and targeting for every vertex is the frontier
"""
from collections.abc import Callable, Iterable, Mapping
import sys
import os
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
from src.type.Subgraph import Subgraph
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from src.framework.GASProgram import GASProgram
from src.framework.strategy.Strategy import Strategy
from src.framework.partition.Partition import Partition
import copy

class ComputeOnGPU(mp.Process):
    def __init__(self, prog: GASProgram, rank, size, result_queue=None, **kwargs) -> None:
        """
        Initinalize the compute process
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.size = size
        self.device = f"cuda:{rank}"
        # 记录本地节点数据, Dense Format
        # self.vertex_data = prog.vertex_data
        self.prog = prog
        self.prog.to(device=self.device)
        self.prog.graph.to(device=self.device)
        self.subgraph = self.prog.graph
        if self.subgraph == None:
            logging.info('subgraph is None')
        num_all_vertex = self.prog.vertex_data.shape[0]
        # 本地节点的掩码 用于本地数据更新 根据掩码 只更新掩码为True的节点
        self.changed_mask = torch.zeros(num_all_vertex, dtype=torch.bool).to(self.device)
        self.changed_mask[self.subgraph.sub_vertices] = True
        # self.prog.set_graph(self.subgraph)
        # 激活节点掩码
        # self.activated_vertex_mask = prog.activated_vertex_mask
        # self.next_activated_vertex_mask = prog.next_activated_vertex_mask
        # device
        # self.device = 'cpu'
        
    def init_porcess(self, backend='nccl'):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=self.rank, world_size=self.size)
    
    def run(self):
        self.init_porcess()
        group = dist.new_group([i for i in range(self.size)])
        # 消除初始化环境的影响
        tensor = torch.tensor([self.rank]).to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        # do computation
        """
        1. 获取本地数据的dense表示
        2. AllReduce 获取全局数据
        3. 计算更新数据
        """
        # viztracer profile 可选：默认关闭，通过环境变量 MTGRAPH_PROFILE=1 开启
        # 用法：MTGRAPH_PROFILE=1 python xxx.py
        # 输出：result<rank>.json（每 rank 一个 trace 文件，可在 perfetto.io 或 vizviewer 中查看）
        tracer = None
        if os.environ.get('MTGRAPH_PROFILE', '0') == '1':
            try:
                from viztracer import VizTracer
                tracer = VizTracer()
                tracer.start()
            except ImportError:
                logging.warning('MTGRAPH_PROFILE=1 但未安装 viztracer，跳过 profile。'
                                '可执行 pip install viztracer 安装。')
        t_begin = time.time()
        communication_time = 0

        # logging.info('procees {} mask {}'.format(self.rank, self.prog.activated_vertex_mask))
        while not torch.all(self.prog.activated_vertex_mask == 0):
            # t1 = time.time()
            logging.info('process {} begin iter {} '.format(self.rank, self.prog.curr_iter))
            self.prog.vertex_data = torch.where(self.changed_mask.unsqueeze(1), self.prog.vertex_data, 
                                                  torch.zeros_like(self.prog.vertex_data))
            t1 = time.time()
            dist.all_reduce(self.prog.vertex_data, op=dist.ReduceOp.SUM, group=group)
            t2 = time.time()
            communication_time += t2 - t1
            logging.info('process {} iter {} all reduce time {}'.format(self.rank,self.prog.curr_iter, t2 - t1))
            # 获取本地激活节点  sub_vertex_id
            """
            没必要转换为 origin_vertex_id,
            但 SubGraph的方法中的方法的参数均为 origin_vertex_id 然后将其转为 sub_vertex_id
            然后进行计算
            """
            local_activated_vertex_list = torch.where(self.prog.activated_vertex_mask[self.subgraph.sub_vertices])[0]
            local_activated_vertex_list = self.subgraph.sub_vertices[local_activated_vertex_list]

            # 本轮迭代本地没有激活节点 直接跳过本轮迭代
            if(local_activated_vertex_list.shape[0] == 0):
                self.prog.curr_iter +=1
                continue
            # do local computation 可以封装为一个函数s

            # gather nbrs 并不是每轮迭代都要执行
            if self.prog.curr_iter == 0:
                self.g_nbrs, self.g_edges, self.g_ptr = self.prog.gather_nbrs(local_activated_vertex_list)
            # logging.info('process {} device {}'.format(self.rank, local_activated_vertex_list.device))
            # gather
            gd, g_ptr = self.prog.gather(local_activated_vertex_list, self.g_nbrs, self.g_edges, self.g_ptr)
            # sum
            gsum = self.prog.sum(gd, g_ptr)
            # apply
            apply_d, apply_mask = self.prog.apply(local_activated_vertex_list, gsum)
            if apply_d is not None and apply_mask is not None:
                self.prog.vertex_data[local_activated_vertex_list] = torch.where(apply_mask, apply_d, self.prog.vertex_data[local_activated_vertex_list])

            # 并不是每轮迭代都要执行
            if self.prog.curr_iter == 0:
                self.s_nbrs, self.s_edges, self.s_ptr = self.prog.scatter_nbrs(local_activated_vertex_list)
            s_data, s_mask = self.prog.scatter(local_activated_vertex_list, self.s_nbrs, self.s_edges, self.s_ptr, apply_d)    

            if not self.prog.not_change_activated:
                self.prog.activated_vertex_mask = self.prog.next_activated_vertex_mask
                self.prog.next_activated_vertex_mask = torch.zeros_like(self.prog.activated_vertex_mask)

                # ALl_Reduce 
                # activated_vertex_mask_tmp = self.prog.activated_vertex_mask.to(torch.int8)
                # dist.all_reduce(activated_vertex_mask_tmp, op=dist.ReduceOp.BOR, group=group)
                # self.prog.activated_vertex_mask = activated_vertex_mask_tmp.to(torch.bool)
                # dist.all_reduce(self.prog.activated_vertex_mask, op=dist.ReduceOp.BOR, group=group)
            else:
                self.prog.not_change_activated = False

            self.prog.curr_iter +=1
            # t2 = time.time()
            # logging.info('process {}  use time {} for iter {}'.format(self.rank, t2 - t1, self.prog.curr_iter)) 
        if tracer is not None:
            tracer.stop()
            tracer.save("result{}.json".format(self.rank))
        t_end = time.time()
        logging.info('process {} all time {} communication time {}'.format(self.rank, t_end - t_begin, communication_time))

class MultiGPUStrategyByNCCL(Strategy):
    def __init__(self, prog: GASProgram, partition: Partition, device_num = 1) -> None:
        super().__init__()
        self.prog = prog
        self.partition = partition
        self.device_num = device_num

    def compute(self):
        mp.set_start_method('spawn', force=True)
        # do partition
        self.partition.set_graph(self.prog.graph)
        self.partition.set_num_partitions(self.device_num)
        partitions = self.partition.generate_partitions()
        subgraphs = []
        

        for (p, i), v in partitions:
            orig_to_sub_vertices = torch.zeros_like(self.prog.graph.vertices)    
            orig_to_sub_vertices[v] = torch.arange(v.numel())
            subgraph = Subgraph(p, v, i, orig_to_sub_vertices)
            subgraphs.append(subgraph)
        
        # store final result by Queue
        queue = mp.Queue(maxsize=1)

        # 避免 deepcopy 时候 重复拷贝 Graph 实例
        self.prog.set_graph(None)

        # init and run processes
        processes = []
        t1 = time.time()
        for rank in range(self.device_num):
            prog = copy.deepcopy(self.prog)
            prog.set_graph(subgraphs[rank])
            # rank 0 process get result so pass queue
            p = ComputeOnGPU(prog, rank, self.device_num, queue) \
                if rank == 0 else ComputeOnGPU(prog, rank, self.device_num)
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        t2 = time.time()
        logging.info(t2 - t1)
        # get result
        # result = queue.get()

        # return result
        return None, None
        

        