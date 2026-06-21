import torch
import torch.distributed as dist
import os
import time


def train(rank, world_size):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    
    print(f'rank {rank}  world_size {world_size}')

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print('The rank in env is: ', rank)
    train(rank, world_size)
    