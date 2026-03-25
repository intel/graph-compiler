from .graph_compiler import *

try:
    import torch

    torch.Tensor.usm = lambda self, shared=False: Usm(self, shared)
except ImportError:
    pass

try:
    from torch import Tensor, nn
    from torch_mlir.fx import OutputType, export_and_import

    torch.Tensor.usm = lambda self, shared=False: Usm(self, shared)

    def to_gc_mod(
        self: nn.Module, *sample_args: Tensor, dump=False, wait=True, **sample_kwargs
    ):
        mlir = export_and_import(
            self,
            *sample_args,
            output_type=OutputType.LINALG_ON_TENSORS,
            **sample_kwargs,
        )
        return GpuModule(str(mlir), dump=dump, wait=wait)

    nn.Module.gc = to_gc_mod

except ImportError:
    pass
