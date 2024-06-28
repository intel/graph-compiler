import json
import os

from gc_mlir.dialects import onednn_graph
from gc_mlir.dialects._ods_common import OpView
from gc_mlir.extras import types as T
from gc_mlir.ir import IntegerAttr
import copy


class Config:
    def __init__(self):
        self.field_candidates = {}
        self.field_constraints = {}
        self.init_candidates()
        self.init_constraints()

    def init_candidates(self):
        pass

    def init_constraints(self):
        pass

    def attach_to_ir(self, op: OpView):
        pass

class MatMulConfig(Config):
    def __init__(
        self,
        M_threads: int = 1,
        K_threads: int = 1,
        N_threads: int = 1,
        M_block: int = 1,
        K_block: int = 1,
        N_block: int = 1,
        innermostM_block: int = 1,
        innermostK_block: int = 1,
        innermostN_block: int = 1,
    ):
        super().__init__()
        self.M_threads = M_threads
        self.K_threads = K_threads
        self.N_threads = N_threads
        self.M_block = M_block
        self.K_block = K_block
        self.N_block = N_block
        self.innermostM_block = innermostM_block
        self.innermostK_block = innermostK_block
        self.innermostN_block = innermostN_block

    def __init__(self, op: OpView):
        super().__init__()
        # assert isinstance(op, onednn_graph.MatMulOp)
        # you can set the default value by matmul_op
        # cpu_counts = os.cpu_count()
        
        print(op.operands[0].type.shape)
        
        # self.input_a_shape = op.input_a.type.shape
        # self.input_b_shape = op.input_b.type.shape
        # self.input_a_dtype = op.input_a.type.element_type
        self.M_threads = 1
        self.K_threads = 1
        self.N_threads = 1
        
        self.M_block = 1
        self.K_block = 1
        self.N_block = 1
        self.innermostM_block = 1
        self.innermostK_block = 1
        self.innermostN_block = 1

    def init_candidates(self):
        # you can set the candidates by info form matmul op
        def get_factors(num: int):
            factors = []
            for i in range(1, num+1):
                if num % i == 0:
                    factors.append(i)
            return factors
        f = get_factors(56)

        self.field_candidates["M_threads"] = copy.deepcopy(f)
        self.field_candidates["K_threads"] = copy.deepcopy(f)
        self.field_candidates["N_threads"] = copy.deepcopy(f)
        self.field_candidates["M_block"] = [16]
        self.field_candidates["K_block"] = [16]
        self.field_candidates["N_block"] = [16]

    def init_constraints(self):
        # example: using lambda to add constraints
        # self.field_constraints["K_block"] = (
        #     lambda MatMulConfig, K_block: MatMulConfig.M_block <= K_block
        # )
        self.field_constraints["M_threads"] = None
        self.field_constraints["K_threads"] = (
            lambda MatMulConfig, K_threads: 56 / MatMulConfig.M_threads % K_threads == 0
        )
        self.field_constraints["N_threads"] = (
            lambda MatMulConfig, N_threads: 56 / MatMulConfig.M_threads / MatMulConfig.K_threads % N_threads == 0
        )
        self.field_constraints["M_block"] = None
        self.field_constraints["K_block"] = None
        self.field_constraints["N_block"] = None

    def attach_to_ir(self, op: OpView):
        # assert isinstance(op, onednn_graph.MatMulOp)
        attr_to_field = {
            "Mthreads": self.M_threads,
            "Kthreads": self.K_threads,
            "Nthreads": self.N_threads,
            "MBlock": self.M_block,
            "KBlock": self.K_block,
            "NBlock": self.N_block,
            "innermostMBlock": self.innermostM_block,
            "innermostKBlock": self.innermostK_block,
            "innermostNBlock": self.innermostN_block,
        }
        for name, value in attr_to_field.items():
            op.attributes[name] = IntegerAttr.get(T.i32(), value)
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        obj_dict = {
            "MatMulConfig": {
                "M_threads": self.M_threads,
                "K_threads": self.K_threads,
                "N_threads": self.N_threads,
                "M_block": self.M_block,
                "K_block": self.K_block,
                "N_block": self.N_block,
                "innermostM_block": self.innermostM_block,
                "innermostK_block": self.innermostK_block,
                "innermostN_block": self.innermostN_block,
            }
        }
        return json.dumps(obj_dict, indent=4)
