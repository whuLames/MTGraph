"""
The class `GASProgram` allows multiple strategies, which define how `GASProgram` performs the GAS operations behind the scene. This file defines interfaces for the strategies.
"""
from abc import *

class Strategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute(self):
        """
        This function defines the detailed GAS computational operations. For example, SimpleStrategy applies no partitioning. The memory scheduler is implemented here.
        """
        raise NotImplementedError()
    
    def set_program(self, program):
        self.program = program
