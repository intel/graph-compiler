# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from compileall import compile_dir

from gc_mlir.graph_compiler import GraphCompiler
from gc_mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


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
