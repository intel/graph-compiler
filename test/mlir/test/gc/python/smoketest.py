# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.dialects import func, onednn_graph
from gc_mlir.graph_compiler import GraphCompiler
from gc_mlir.ir import *


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


# CHECK-LABEL: TEST: testCompiler
@run
def testCompiler():
    with Context():
        module = Module.parse(
            """
            func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>) -> tensor<128x256xf32> {
                %0 = onednn_graph.matmul %arg0, %arg1 : (tensor<128x512xf32>, tensor<512x256xf32>) -> tensor<128x256xf32>
                return %0 : tensor<128x256xf32>
            }
            """
        )

        compiler = GraphCompiler(
            pipeline="builtin.module(convert-onednn-graph-to-linalg)"
        )
        compiler.compile(module)
        #  CHECK-NOT: onednn_graph.matmul
        print(module)
