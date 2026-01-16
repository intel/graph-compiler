from .graph_compiler import *

try:
    import torch
    torch.Tensor.usm = lambda self, shared=False: Usm(self, shared)
except ImportError:
    pass
