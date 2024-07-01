# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from compileall import compile_dir

import gc_mlir
from gc_mlir.graph_compiler import GraphCompiler
from gc_mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testCompiler
@run
def testCompiler():
    with Context() as ctx:
        ctx.allow_unregistered_dialects=True
        module = Module.parse(
            """
            module {
  func.func @entry(%arg0: tensor<128x128x32x32xbf16>, %arg1: tensor<128x128x16x32x2xbf16>) -> tensor<128x128x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x128x32x32xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16>
    %2 = linalgx.mm4d_vnni ins(%arg0, %arg1 : tensor<128x128x32x32xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16>
    return %2 : tensor<128x128x32x32xbf16>
  }
}
          """
        )

        # compiler = GraphCompiler(pipeline=" any(convert-onednn-graph-to-linalg,print-ir{label=Frontend passes result},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,func.func(deep-tile-contraction-named-op),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=Tensor passes result},func.func(arith-expand{include-bf16=false}),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=Vector passes result},one-shot-bufferize{allow-return-allocs-from-loops=false allow-unknown-ops=false analysis-fuzzer-seed=0 analysis-heuristic=bottom-up bufferize-function-boundaries=false check-parallel-regions=true copy-before-write=false  dump-alias-sets=false function-boundary-type-conversion=infer-layout-map must-infer-memory-space=false  print-conflicts=false test-analysis-only=false unknown-type-conversion=fully-dynamic-layout-map},cse,buffer-results-to-out-params{add-result-attr=false hoist-static-allocs=false},func.func(buffer-hoisting),func.func(buffer-loop-hoisting),func.func(buffer-deallocation),convert-bufferization-to-memref,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=Bufferization passes result},func.func(convert-linalg-to-microkernel),early-dispatch-microkernel,convert-microkernel-to-dnnl-func,merge-branch-microkernel-context,microkernel-invariant-code-motion,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=MicroKernel passes result},linalg-lower-to-loop,scf-forall-to-parallel,convert-scf-to-openmp{num-threads=0},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=CPURuntime passes result},memref-expand,expand-strided-metadata,lower-affine,finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},convert-scf-to-cf,convert-cpuruntime-to-llvm,convert-openmp-to-llvm,func.func(convert-math-to-llvm{approximate-log1p=true}),convert-math-to-libm,func.func(convert-arith-to-llvm{index-bitwidth=0}),convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},convert-cf-to-llvm{index-bitwidth=0},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},reconcile-unrealized-casts,symbol-dce,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=LoweringToLLVM passes result},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,loop-invariant-code-motion,cse,sccp,print-ir{label=LLVM passes result})")
        
        compiler = GraphCompiler()
        compiler.compile(module)

        #  CHECK-NOT: onednn_graph.matmul
        print(module)
