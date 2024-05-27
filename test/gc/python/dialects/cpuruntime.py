# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.ir import *
from gc_mlir.dialects import cpuruntime, func


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testCPURuntimeOps
@run
def testCPURuntimeOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(F32Type.get(), IntegerType.get_signless(32))
            def do_print(arg1, arg2):
                cpuruntime.printf("Hello world %f %d", [arg1, arg2])
                return

        # CHECK-LABEL: func @do_print(
        # CHECK-SAME:                  %[[ARG_0:.*]]: f32, %[[ARG_1:.*]]: i32) {
        # CHECK:         cpuruntime.printf "Hello world %f %d" %[[ARG_0]], %[[ARG_1:.*]] : f32, i32
        # CHECK:         return
        # CHECK:       }
        print(module)

