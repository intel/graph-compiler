// RUN: gc-opt %s --pass-pipeline='builtin.module(func.func(iterative-tiling-and-fusion{use-cost-model=0 default-tile-size=matmul:{16,16}}),eliminate-empty-tensors,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},drop-equivalent-buffer-results,func.func(finalizing-bufferize),canonicalize,cse,drop-equivalent-buffer-results,expand-realloc,canonicalize,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref,func.func(scf-forall-to-parallel),func.func(linalg-to-xegpu{stages=1 dpas-tile=8,16,16 k-tile=16}),xegpu-fold-alias-ops,func.func(convert-linalg-to-parallel-loops),func.func(gpu-map-parallel-loops),func.func(convert-parallel-loops-to-gpu),func.func(insert-gpu-allocs),gpu-kernel-outlining,canonicalize,set-spirv-capabilities{client-api=opencl},gpu.module(set-spirv-abi-attrs{client-api=opencl}),lower-affine,imex-vector-linearize,gpu.module(convert-xegpu-to-vc),reconcile-unrealized-casts,bf16-to-gpu,gpu.module(convert-func-to-spirv),gpu.module(convert-vector-to-spirv),imex-convert-gpu-to-spirv,spirv.module(spirv-lower-abi-attrs,spirv-update-vce),func.func(llvm-request-c-wrappers),serialize-spirv,convert-vector-to-scf,convert-gpu-to-gpux,convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,convert-math-to-llvm,convert-gpux-to-llvm,convert-index-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | imex-cpu-runner -e main --entry-point-result=void \
// RUN:   --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime | FileCheck %s
module{

memref.global "private" @__constant_512x512xf16 : memref<512x512xf16> = dense<0.0>

func.func @linalg_matmul(%arg0: tensor<512x512xf16>,
                 %arg1: tensor<512x512xf16>,
                 %arg2: tensor<512x512xf16>) -> tensor<512x512xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf16>, tensor<512x512xf16>)
                     outs(%arg2 : tensor<512x512xf16>) -> tensor<512x512xf16>
  return %0 : tensor<512x512xf16>
}

func.func @generate_t(%div : f16) -> tensor<512x512xf16> {
    %c32 = arith.constant 512.0 : f16
    %c10 = arith.constant 10.0 : f16

    %0 = tensor.generate {
      ^bb0(%i : index, %j : index):
        %cst32 = arith.constant 512.0 : f16
        %int0 = arith.index_cast %i : index to i16
        %int1 = arith.index_cast %j : index to i16
        %fp1 = arith.uitofp %int0 : i16 to f16
        %fp2 = arith.uitofp %int1 : i16 to f16

        // %tmp1 = arith.mulf %fp1, %cst32 : f16
        %tmp2 = arith.addf %fp1, %fp2 : f16
        %res = arith.divf %tmp2, %div : f16

        // %tmp2 = arith.mulf %res, %step : f16
        // %val = arith.addf %min, %tmp2 : f16
        tensor.yield %res : f16
    } : tensor<512x512xf16>
    return %0 : tensor<512x512xf16>
}

func.func @cpu_matmul(%a : tensor<512x512xf16>, %b : tensor<512x512xf16>) -> memref<512x512xf16> {
  %ref = memref.get_global @__constant_512x512xf16 : memref<512x512xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 512 : index

  // scf.for won't be parallelized/mapped to GPU thus this code will be executed on CPU
  scf.for %arg0 = %c0 to %c32 step %c1 {
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %acc = memref.load %ref[%arg0, %arg1] : memref<512x512xf16>
      %accf32 = arith.extf %acc : f16 to f32
      %res = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %accf32) -> f32 {
        %ai = tensor.extract %a[%arg0, %arg2] : tensor<512x512xf16>
        %bi = tensor.extract %b[%arg2, %arg1] : tensor<512x512xf16>
        %c = arith.mulf %ai, %bi : f16
        %cc = arith.extf %c : f16 to f32
        %ccc = arith.addf %cc, %arg3 : f32
        scf.yield %ccc : f32
      }
      %res16 = arith.truncf %res : f32 to f16
      memref.store %res16, %ref[%arg0, %arg1] : memref<512x512xf16>
    }
  }

  return %ref : memref<512x512xf16>
}

func.func @main() {
  %a0 = arith.constant 100.0 : f16
  %0 = call @generate_t(%a0) : (f16) -> tensor<512x512xf16>

  %a1 = arith.constant 200.0 : f16
  %1 = call @generate_t(%a1) : (f16) -> tensor<512x512xf16>

  %3 = call @cpu_matmul(%0, %1) : (tensor<512x512xf16>, tensor<512x512xf16>) -> memref<512x512xf16>
  // %unranked = tensor.cast %3 : tensor<512x512xf16> to tensor<*xf16>
  // call @printMemrefF16(%unranked) : (tensor<*xf16>) -> ()
  %2 = arith.constant dense<0.0> : tensor<512x512xf16>
  %4 = call @linalg_matmul(%0, %1, %2) : (tensor<512x512xf16>, tensor<512x512xf16>, tensor<512x512xf16>) -> tensor<512x512xf16>
  // %unranked = memref.cast %3 : memref<512x512xf16> to memref<*xf16>
  // call @printMemrefF16(%unranked) : (memref<*xf16>) -> ()

  %cast = tensor.cast %4 : tensor<512x512xf16> to tensor<*xf16> 
  // call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()
  // %cast_ref = memref.cast %3 : memref<512x512xf16> to memref<*xf16>
  call @printAllcloseF16(%cast, %cast) : (tensor<*xf16>, tensor<*xf16>) -> ()
  return
}

func.func private @printMemrefF16(%ptr : tensor<*xf16>)
func.func private @printAllcloseF16(tensor<*xf16>, tensor<*xf16>)
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [32, 32] strides = [32, 1] data =
// CHECK-NEXT: [815,   816.5,   817.5,   819,   820,   821.5,   822.5,   824,   825,   826,   827,   828.5,   830,   831,   832,   833.5,   834.5,   836,   837,   838.5,   839.5,   840.5,   841.5,   843.5,   844.5,   845.5,   846.5,   848,   849,   850.5,   851.5,   853], 
// CHECK-NEXT: [2058,   2062,   2064,   2068,   2072,   2076,   2080,   2084,   2088,   2090,   2094,   2098,   2102,   2106,   2110,   2114,   2116,   2120,   2124,   2128,   2132,   2136,   2138,   2144,   2146,   2150,   2154,   2158,   2162,   2166,   2168,   2172], 
// CHECK-NEXT: [3298,   3304,   3310,   3318,   3324,   3330,   3336,   3342,   3348,   3354,   3360,   3368,   3374,   3380,   3386,   3392,   3398,   3404,   3410,   3418,   3424,   3430,   3434,   3442,   3448,   3454,   3460,   3468,   3472,   3478,   3484,   3492], 
// CHECK-NEXT: [4544,   4552,   4560,   4568,   4576,   4584,   4592,   4604,   4612,   4620,   4628,   4640,   4648,   4656,   4664,   4672,   4680,   4692,   4700,   4708,   4716,   4724,   4732,   4744,   4752,   4760,   4768,   4780,   4788,   4796,   4804,   4812], 
// CHECK-NEXT: [5784,   5796,   5804,   5816,   5828,   5840,   5848,   5864,   5872,   5884,   5896,   5908,   5916,   5928,   5940,   5952,   5964,   5972,   5984,   5996,   6008,   6020,   6028,   6044,   6052,   6064,   6072,   6088,   6096,   6108,   6120,   6132], 
// CHECK-NEXT: [7024,   7036,   7052,   7068,   7080,   7092,   7104,   7120,   7136,   7148,   7160,   7176,   7188,   7204,   7216,   7232,   7244,   7256,   7272,   7284,   7300,   7312,   7324,   7340,   7352,   7368,   7380,   7396,   7408,   7420,   7436,   7452], 
// CHECK-NEXT: [8272,   8288,   8304,   8320,   8336,   8352,   8368,   8384,   8400,   8416,   8432,   8448,   8464,   8480,   8496,   8512,   8528,   8544,   8560,   8576,   8592,   8608,   8624,   8648,   8656,   8672,   8688,   8712,   8728,   8744,   8752,   8776], 
// CHECK-NEXT: [9512,   9528,   9544,   9568,   9584,   9608,   9624,   9640,   9664,   9680,   9696,   9720,   9736,   9752,   9768,   9792,   9808,   9832,   9848,   9872,   9888,   9904,   9920,   9944,   9960,   9976,   10000,   10016,   10032,   10056,   10072,   10096], 
// CHECK-NEXT: [10752,   10776,   10792,   10816,   10840,   10856,   10880,   10904,   10920,   10944,   10960,   10984,   11008,   11024,   11048,   11072,   11088,   11112,   11136,   11160,   11176,   11200,   11216,   11240,   11264,   11280,   11304,   11328,   11344,   11368,   11384,   11408], 
// CHECK-NEXT: [11992,   12016,   12040,   12064,   12088,   12112,   12136,   12160,   12184,   12208,   12232,   12256,   12280,   12304,   12320,   12352,   12376,   12400,   12416,   12448,   12464,   12488,   12512,   12544,   12560,   12584,   12608,   12632,   12656,   12680,   12704,   12728], 
// CHECK-NEXT: [13232,   13256,   13288,   13312,   13336,   13368,   13392,   13416,   13440,   13472,   13496,   13528,   13552,   13576,   13600,   13632,   13656,   13680,   13704,   13736,   13760,   13784,   13808,   13840,   13864,   13888,   13912,   13944,   13968,   13992,   14016,   14048], 
// CHECK-NEXT: [14472,   14504,   14528,   14560,   14592,   14616,   14648,   14680,   14704,   14736,   14760,   14792,   14816,   14848,   14872,   14904,   14936,   14960,   14992,   15024,   15048,   15080,   15104,   15136,   15168,   15192,   15216,   15256,   15280,   15304,   15336,   15368], 
// CHECK-NEXT: [15728,   15760,   15784,   15824,   15848,   15880,   15912,   15944,   15976,   16008,   16032,   16072,   16104,   16128,   16160,   16200,   16224,   16256,   16288,   16320,   16352,   16384,   16416,   16448,   16480,   16512,   16528,   16576,   16608,   16624,   16656,   16704], 
// CHECK-NEXT: [16960,   16992,   17024,   17072,   17104,   17136,   17168,   17200,   17232,   17264,   17296,   17344,   17376,   17408,   17440,   17472,   17504,   17536,   17568,   17616,   17648,   17680,   17712,   17744,   17776,   17808,   17840,   17888,   17904,   17952,   17984,   18016], 
// CHECK-NEXT: [18208,   18240,   18272,   18320,   18352,   18384,   18416,   18464,   18496,   18528,   18560,   18608,   18640,   18672,   18720,   18752,   18784,   18816,   18864,   18896,   18928,   18976,   19008,   19040,   19072,   19120,   19152,   19184,   19216,   19264,   19296,   19328], 
// CHECK-NEXT: [19456,   19488,   19520,   19568,   19600,   19648,   19680,   19728,   19760,   19792,   19840,   19872,   19920,   19952,   19984,   20032,   20064,   20112,   20144,   20192,   20224,   20256,   20304,   20336,   20384,   20416,   20448,   20496,   20528,   20576,   20608,   20656], 
// CHECK-NEXT: [20688,   20736,   20768,   20816,   20848,   20896,   20928,   20976,   21024,   21056,   21104,   21152,   21184,   21232,   21264,   21312,   21344,   21392,   21424,   21472,   21520,   21552,   21600,   21648,   21680,   21728,   21760,   21808,   21840,   21888,   21920,   21968], 
// CHECK-NEXT: [21936,   21968,   22016,   22064,   22112,   22144,   22192,   22240,   22288,   22320,   22368,   22416,   22448,   22496,   22544,   22592,   22624,   22672,   22720,   22768,   22800,   22848,   22896,   22944,   22976,   23024,   23072,   23120,   23152,   23200,   23232,   23296], 
// CHECK-NEXT: [23168,   23216,   23264,   23312,   23360,   23408,   23440,   23504,   23536,   23584,   23632,   23680,   23728,   23776,   23808,   23872,   23904,   23952,   24000,   24048,   24096,   24144,   24192,   24240,   24288,   24320,   24368,   24416,   24464,   24512,   24560,   24608], 
// CHECK-NEXT: [24416,   24464,   24512,   24560,   24608,   24656,   24704,   24752,   24800,   24848,   24896,   24944,   24992,   25040,   25088,   25152,   25200,   25248,   25280,   25344,   25392,   25440,   25488,   25536,   25584,   25632,   25680,   25728,   25776,   25824,   25872,   25920], 
// CHECK-NEXT: [25648,   25712,   25760,   25808,   25856,   25904,   25952,   26016,   26064,   26112,   26160,   26224,   26272,   26320,   26368,   26432,   26480,   26528,   26576,   26624,   26672,   26736,   26784,   26832,   26880,   26928,   26976,   27040,   27088,   27136,   27184,   27248], 
// CHECK-NEXT: [26896,   26944,   26992,   27056,   27104,   27168,   27216,   27280,   27328,   27376,   27424,   27488,   27536,   27600,   27648,   27712,   27760,   27808,   27856,   27920,   27968,   28016,   28080,   28128,   28192,   28240,   28288,   28352,   28400,   28448,   28496,   28560], 
// CHECK-NEXT: [28128,   28192,   28240,   28304,   28368,   28416,   28464,   28528,   28592,   28640,   28688,   28752,   28816,   28864,   28912,   28976,   29040,   29088,   29152,   29216,   29264,   29312,   29376,   29440,   29488,   29536,   29600,   29664,   29712,   29760,   29824,   29888], 
// CHECK-NEXT: [29376,   29440,   29488,   29552,   29616,   29664,   29728,   29792,   29840,   29904,   29952,   30032,   30080,   30144,   30192,   30256,   30320,   30368,   30432,   30496,   30544,   30608,   30672,   30736,   30784,   30848,   30896,   30960,   31024,   31072,   31136,   31200], 
// CHECK-NEXT: [30640,   30704,   30752,   30832,   30880,   30944,   31008,   31072,   31120,   31184,   31248,   31312,   31376,   31440,   31488,   31568,   31616,   31680,   31728,   31808,   31856,   31920,   31984,   32048,   32112,   32176,   32224,   32288,   32352,   32416,   32464,   32544], 
// CHECK-NEXT: [31872,   31936,   32000,   32080,   32128,   32192,   32256,   32336,   32384,   32448,   32512,   32576,   32640,   32704,   32768,   32832,   32896,   32960,   33024,   33088,   33152,   33216,   33280,   33344,   33408,   33472,   33536,   33600,   33664,   33728,   33792,   33856], 
// CHECK-NEXT: [33120,   33184,   33248,   33312,   33376,   33440,   33504,   33600,   33664,   33728,   33792,   33856,   33920,   33984,   34048,   34112,   34176,   34240,   34304,   34368,   34432,   34496,   34560,   34656,   34720,   34784,   34848,   34912,   34976,   35040,   35104,   35168], 
// CHECK-NEXT: [34368,   34432,   34496,   34560,   34624,   34688,   34752,   34848,   34912,   34976,   35040,   35136,   35200,   35264,   35328,   35392,   35456,   35520,   35584,   35680,   35744,   35808,   35872,   35936,   36000,   36064,   36128,   36224,   36288,   36352,   36416,   36512], 
// CHECK-NEXT: [35616,   35680,   35744,   35808,   35872,   35968,   36032,   36096,   36160,   36256,   36320,   36384,   36448,   36512,   36608,   36672,   36736,   36800,   36864,   36960,   37024,   37088,   37152,   37248,   37312,   37376,   37440,   37536,   37600,   37664,   37728,   37824], 
// CHECK-NEXT: [36832,   36928,   36992,   37056,   37152,   37216,   37280,   37376,   37440,   37504,   37568,   37664,   37728,   37792,   37856,   37952,   38016,   38080,   38176,   38240,   38304,   38400,   38464,   38560,   38624,   38688,   38752,   38848,   38912,   38976,   39040,   39136], 
// CHECK-NEXT: [38080,   38144,   38240,   38304,   38400,   38464,   38528,   38624,   38688,   38784,   38848,   38912,   39008,   39072,   39136,   39232,   39296,   39392,   39456,   39552,   39616,   39680,   39744,   39840,   39904,   40000,   40064,   40160,   40224,   40288,   40352,   40448], 
// CHECK-NEXT: [39328,   39392,   39488,   39552,   39648,   39712,   39776,   39872,   39968,   40032,   40096,   40192,   40256,   40352,   40416,   40512,   40576,   40672,   40736,   40832,   40896,   40992,   41056,   41152,   41216,   41280,   41376,   41472,   41536,   41600,   41696,   41760]
