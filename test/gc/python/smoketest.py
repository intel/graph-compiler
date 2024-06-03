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


# CHECK-LABEL: TEST: testCreateOp
# CHECK onednn_graph.add
@run
def testCreateOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32 = F32Type.get(ctx)
        tensor_type = RankedTensorType.get([128, 256], f32)
        with InsertionPoint(module.body):
            f = func.FuncOp(
                name="add",
                type=FunctionType.get(
                    inputs=[tensor_type, tensor_type], results=[tensor_type]
                ),
            )
            with InsertionPoint(f.add_entry_block()):
                arg0, arg1 = f.entry_block.arguments
                result = onednn_graph.AddOp(arg0, arg1).result
                func.ReturnOp([result])
        print(module)


# CHECK-LABEL: TEST: testPassManager
@run
def testPassManager():
    with Context():
        module = Module.parse(
            """
            // CHECK: [[C0:%.+]] = arith.constant 0
            // CHECK: [[INIT:%.+]] = tensor.empty()
            // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
            // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
            func.func @matmul(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>) -> tensor<128x256xbf16> {
                %0 = onednn_graph.matmul %arg0, %arg1 : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>
                return %0 : tensor<128x256xbf16>
            }
            """
        )
        pm = PassManager.parse("builtin.module(convert-onednn-graph-to-linalg)")
        pm.run(module.operation)
        print(module)
