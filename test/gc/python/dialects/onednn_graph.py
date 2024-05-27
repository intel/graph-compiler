# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.ir import *
from gc_mlir.dialects import onednn_graph, func


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
