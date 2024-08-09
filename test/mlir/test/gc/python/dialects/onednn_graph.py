# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.ir import *
from gc_mlir.dialects import onednn_graph, func
from gc_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testOneDNNGraphOps
@run
def testOneDNNGraphOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f32 = F32Type.get(ctx)
            tensor_type = RankedTensorType.get([128, 128], f32)

            @func.FuncOp.from_py_func(tensor_type, tensor_type)
            def entry(arg1, arg2):
                res1 = onednn_graph.matmul(
                    arg1, arg2, bias=None, transpose_a=False, transpose_b=False
                )
                res2 = onednn_graph.add(res1, arg2)
                return onednn_graph.relu(res2)

        # CHECK: [[MM:%.+]] = onednn_graph.matmul
        # CHECK: [[ADD:%.+]] = onednn_graph.add
        # CHECK: [[RELU:%.+]] = onednn_graph.relu
        print(module)


# CHECK-LABEL: TEST: testConvertToLinalg
@run
def testConvertToLinalg():
    with Context():
        module = Module.parse(
            """
            func.func @matmul(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>) -> tensor<128x256xbf16> {
                %0 = onednn_graph.matmul %arg0, %arg1 : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>
                return %0 : tensor<128x256xbf16>
            }
            """
        )
        pm = PassManager.parse("builtin.module(convert-onednn-graph-to-linalg)")
        # CHECK: [[C0:%.+]] = arith.constant 0
        # CHECK: [[INIT:%.+]] = tensor.empty()
        # CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
        # CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
        pm.run(module.operation)
        print(module)
