from driver import *
from gc_mlir import ir
from utils import get_kernel_func_from_module, make_tensor


class LoadMLIRParams(DriverParams):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument("--entry", type=str, default="main_entry")

    def __init__(self, params: dict):
        self.path = params["path"]
        self.main_entry = params["entry"]


class LoadMLIR(Driver):
    def __init__(self, ctx: ir.Context, params: dict):
        self.params = LoadMLIRParams(params)
        self.main_entry = self.params.main_entry
        super().__init__(ctx)

    def _get_mlir(self):
        with open(self.params.path, "r") as file:
            content = file.read()
        return content

    def init_module(self, ctx: ir.Context) -> ir.Module:
        module = ir.Module.parse(self._get_mlir(), ctx)
        return module

    def prepare_np_args(self) -> "list[np.ndarray]":
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_tensor(arg.type))
        return np_args

    def prepare_np_res(self) -> "list[np.ndarray]":
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_res = []
        for res in bench_func.type.results:
            np_res.append(make_tensor(res))
        return np_res
