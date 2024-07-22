# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testParseLinalgxOps
@run
def testParseLinalgxOps():
    with Context():
        module = Module.parse(
            """
            func.func @sigmoid(%arg0: tensor<4x256x64xbf16>, %arg1: tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16> {
                // CHECK: linalgx.sigmoid
                %0 = linalgx.sigmoid ins(%arg0 : tensor<4x256x64xbf16>) outs(%arg1 : tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16>
                return %0 : tensor<4x256x64xbf16>
            }
            """
        )
        print(module)
