"""
A simple strategy that uses one process + one thread all the time.

单机单卡策略, 不涉及 CPU-GPU之间的换入换出
"""

from . import Strategy
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
import time

class SimpleStrategy(Strategy.Strategy):
    def __init__(self, program) -> None:
        super().__init__()
        self.program = program
    
    def compute_oneiter(self, activated_vertex_mask, curr_iter):
        prog = self.program
        nbr_update_freq = prog.nbr_update_freq
        logging.info('begin {} iter'.format(curr_iter))
        # curr_iter 为 0 即开始第一轮迭代时, 或当前迭代轮次是 `nbr_update_freq`的倍数时, 更新 `gather_nbrs`
        if curr_iter == 0 or (nbr_update_freq != 0 and curr_iter % nbr_update_freq == 0):
            self.activated_vertex_list = torch.masked_select(prog.graph.vertices, activated_vertex_mask)
            self.g_nbrs, self.g_edges, self.g_ptr = prog.gather_nbrs(self.activated_vertex_list)
        # gather data
        gd, gd_ptr = prog.gather(self.activated_vertex_list, self.g_nbrs, self.g_edges, self.g_ptr)
        # get 
        gsum = prog.sum(gd, gd_ptr)
        apply_d, apply_mask = prog.apply(self.activated_vertex_list, gsum)
        if apply_d is not None and apply_mask is not None:
            prog.vertex_data[self.activated_vertex_list] = torch.where \
                        (apply_mask, apply_d, prog.vertex_data[self.activated_vertex_list])
        # 同上, 更新 `scatter_nbrs`
        if curr_iter == 0 or (nbr_update_freq != 0 and curr_iter % nbr_update_freq == 0):
            # scatter
            self.s_nbrs, self.s_edges, self.s_ptr = prog.scatter_nbrs(self.activated_vertex_list)
        s_data, s_mask = prog.scatter(self.activated_vertex_list, self.s_nbrs, self.s_edges, self.s_ptr, apply_d)
        if s_data is not None and s_mask is not None:
            prog.edge_data[self.s_edges] = torch.where(s_mask, s_data, prog.edge_data[self.s_edges])
        
    def compute(self):
        """
        SimpleStrategy的计算逻辑
        """
        print('compute')
        prog = self.program
        print('prog.device: {}'.format(prog.device))
        prog.to(prog.device)
        prog.graph.to(prog.device)

        
        # run GAS on all vertices
        while not torch.all(prog.activated_vertex_mask == 0):
            self.compute_oneiter(prog.activated_vertex_mask, prog.curr_iter)
            prog.curr_iter += 1
            if not prog.not_change_activated:
                prog.activated_vertex_mask = prog.next_activated_vertex_mask
                prog.next_activated_vertex_mask = torch.zeros_like(prog.activated_vertex_mask, dtype=torch.bool)
            else:
                prog.not_change_activated = False
                continue
        
        # while not prog.is_quit:
        #     self.compute_oneiter(prog.activated_vertex_mask, prog.curr_iter)
        #     prog.curr_iter += 1

            # # 咱不考虑激活节点的更新
            # if not prog.not_change_activated:
            #     prog.activated_vertex_mask = prog.next_activated_vertex_mask
            #     prog.next_activated_vertex_mask = torch.zeros_like(prog.activated_vertex_mask, dtype=torch.bool)
            # else:
            #     prog.not_change_activated = False
            #     continue
        # self.compute_oneiter(prog.activated_vertex_mask, prog.curr_iter)
        return prog.vertex_data, prog.edge_data
