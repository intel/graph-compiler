# Tuner - auto tuning tools
## Description
Tuner is a tool used to  tuner is a tool used to select the best-performing configuration for a graph with tunable operations. Tunable operations refer to operations, such as matmul, conv, etc., whose kernel performance depends on certain configurations, and a tuner generates different configuration combinations for a graph and records their performance. 

## Prerequisite
`mode T` for benchgc

## Options
Since bench is also required within the tuner, the tuner also supports benchmarking options.
Unlike bench mode, in tuner mode, a batch quantity of modules is generated each time, and The default values for warm-up and repeat have been adjusted accordingly.
* --bench_kind [py, grid]
* --warm_up [int], default: 2
* --repeat [int], default: 4

### --tuning_batch [int]
* The batch size of configs, default: `50`
* The tuner first generates a batch of configurations, then proceeds to perform performance testing on these configs. 

### --early_stop [int]
* If the tuner does not find a better performance after testing the number of configurations specified by the `early_stop` value, it will terminate its execution.
* default: `-1`, represents that early stopping is disabled.

### --max_tuning_iters [int]
* The maximum number of configurations the tuner needs to attempt.
* default: `sys.maxsize`

### --timeout [int]
* The maximum runtime limit for the tuner, unit: second
* default: `-1`, means there is no limit.

### --space_percent [float]
* For the set of all possible configurations for a graph, we refer to it as the tuning space. The value of `space_percent` represents the percentage of configurations that we need to tune.
* value range `(0, 1]`, default: 1.0, means 100 percent of tuning space

### --checkpoint_path [str]
* When the checkpoint file exists, the tuner will first load the contents of the checkpoint to restore the previous state upon startup, and it will update the checkpoint file after executing each batch.

### --search_alg [str]
* There are two algorithms within the tuner to search for new configurations.
* grid: grid search which is a exhaustive search
* ga: genetic algorithm.
* default: `grid`

### Options when `--search_alg ga`
* --ga_random_seed [int]: random seed in genetic algorithm, default: 0
* --ga_elite_num [int]: default: 9
* --ga_mutation_prob [float]: default: 0.1 
* --ga_expected_tune_num [int] : default: 0, In the tuner implemented with a genetic algorithm, a data structure is needed to determine whether a new config is a duplicate of a previous one. By default, a set is used for this purpose when this option is not specified. If the user sets this value, a bloom filter is used instead.

## OP config
If users need to make adjustments to the candidates in the config of tunable operations, please manually modify `op_config.py`.For example, you can reduce the tuning space by adjusting the candidates.

## Skip the tuner for the specified OP

If you need to skip the tuner for certain operations, you can add the following attribute to them in MLIR.
Then you can proceed with tuning by using the `--driver=mlir` option
```
linalg.matmul {skipTuner = true} ins(..) outs(...) ...
```

## Example
* General cmd
```
OMP_NUM_THREADS=1 python3 -m benchgc  --mode T  --driver linalg --case matmul --md 0:128x128xf32 --md 1:128x128xf32 --md 2:128x128xf32 --bench_kind wrapper --wram_up 2 --repeat 2 --search_alg grid --tunning_batch 100 --early_stop 1000 --max_tuning_iters 1000000 --timeout 1000000 --space_percent 0.8 --checkpoint_path {path_to_checkpoint_file}
```

* single matmul
```
OMP_NUM_THREADS=1 python3 -m benchgc  --mode T  --driver linalg --case matmul --md 0:128x128xf32 --md 1:128x128xf32 --md 2:128x128xf32

[ 50 / 512 ] skipped: 79 best: 0.025305896997451782 ms
[ 100 / 512 ] skipped: 105 best: 0.025296583771705627 ms
[ 150 / 512 ] skipped: 115 best: 0.025296583771705627 ms
[ 200 / 512 ] skipped: 135 best: 0.025292858481407166 ms
[ 250 / 512 ] skipped: 147 best: 0.025292858481407166 ms
[ 300 / 512 ] skipped: 165 best: 0.025292858481407166 ms
[ 343 / 512 ] skipped: 169 best: 0.025292858481407166 ms
Tuner returns empty batch, early stop now
Tuning ends in 26.26677966117859 s
Best cost: 0.025292858481407166 ms
Best config: [{
    "MatMulConfig": {
        "MThreads": 1,
        "KThreads": 1,
        "NThreads": 1,
        "MBlock": 128,
        "KBlock": 64,
        "NBlock": 16,
        "innerMostMBlock": 32,
        "innerMostKBlock": 16,
        "innerMostNBlock": 16
    }
}]
mlir:
 module attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : ui32>, #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : ui64>, #dlti.dl_entry<"L3_cache_size_in_bytes", 110100480 : ui64>, #dlti.dl_entry<"num_threads", 1 : i32>, #dlti.dl_entry<"max_vector_width", 512 : i64>>>} {
  func.func @entry(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.matmul {KBlock = 64 : i32, KThreads = 1 : i32, MBlock = 128 : i32, MThreads = 1 : i32, NBlock = 16 : i32, NThreads = 1 : i32, cast = #linalg.type_fn<cast_signed>, innerMostKBlock = 16 : i32, innerMostMBlock = 32 : i32, innerMostNBlock = 16 : i32} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}
```

* mlp

```
OMP_NUM_THREADS=1 python -m benchgc --mode T --driver pattern --case mlp --batch_size=32 --hidden_size_list=16x32x64 --has_bias=1x1 --act_type=relu --warm_up 2 --repeat 2
[ 50 / 1536 ] skipped: 352 best: 0.0069122761487960815 ms
[ 100 / 1536 ] skipped: 415 best: 0.006860122084617615 ms
[ 150 / 1536 ] skipped: 662 best: 0.006856396794319153 ms
[ 200 / 1536 ] skipped: 821 best: 0.006856396794319153 ms
[ 250 / 1536 ] skipped: 972 best: 0.006856396794319153 ms
[ 300 / 1536 ] skipped: 1029 best: 0.006856396794319153 ms
[ 350 / 1536 ] skipped: 1080 best: 0.006834045052528381 ms
[ 400 / 1536 ] skipped: 1131 best: 0.006834045052528381 ms
[ 405 / 1536 ] skipped: 1131 best: 0.006834045052528381 ms
Tuner returns empty batch, early stop now
Tuning ends in 80.10290145874023 s
Best cost: 0.006632879376411438 ms
Best config: [{
    "MatMulConfig": {
        "MThreads": 1,
        "KThreads": 1,
        "NThreads": 1,
        "MBlock": 32,
        "KBlock": 16,
        "NBlock": 32,
        "innerMostMBlock": 32,
        "innerMostKBlock": 16,
        "innerMostNBlock": 16
    }
}, {
    "MatMulConfig": {
        "MThreads": 1,
        "KThreads": 1,
        "NThreads": 1,
        "MBlock": 32,
        "KBlock": 32,
        "NBlock": 16,
        "innerMostMBlock": 16,
        "innerMostKBlock": 32,
        "innerMostNBlock": 16
    }
}]
mlir:
 module attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : ui32>, #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : ui64>, #dlti.dl_entry<"L3_cache_size_in_bytes", 110100480 : ui64>, #dlti.dl_entry<"num_threads", 1 : i32>, #dlti.dl_entry<"max_vector_width", 512 : i64>>>} {
  func.func @entry(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<32x64xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64xf32>) -> tensor<32x64xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = linalg.matmul {KBlock = 16 : i32, KThreads = 1 : i32, MBlock = 32 : i32, MThreads = 1 : i32, NBlock = 32 : i32, NThreads = 1 : i32, cast = #linalg.type_fn<cast_signed>, innerMostKBlock = 16 : i32, innerMostMBlock = 32 : i32, innerMostNBlock = 16 : i32} ins(%arg0, %arg1 : tensor<32x16xf32>, tensor<16x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%arg3 : tensor<32xf32>) outs(%3 : tensor<32x32xf32>) dimensions = [0] 
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.add ins(%2, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = linalg.max ins(%5, %cst_0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = tensor.empty() : tensor<32x64xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %10 = linalg.matmul {KBlock = 32 : i32, KThreads = 1 : i32, MBlock = 32 : i32, MThreads = 1 : i32, NBlock = 16 : i32, NThreads = 1 : i32, cast = #linalg.type_fn<cast_signed>, innerMostKBlock = 32 : i32, innerMostMBlock = 16 : i32, innerMostNBlock = 16 : i32} ins(%7, %arg2 : tensor<32x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %11 = tensor.empty() : tensor<32x64xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg4 : tensor<64xf32>) outs(%11 : tensor<32x64xf32>) dimensions = [0] 
    %12 = tensor.empty() : tensor<32x64xf32>
    %13 = linalg.add ins(%10, %broadcasted_1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%12 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    %14 = tensor.empty() : tensor<32x64xf32>
    %15 = linalg.max ins(%13, %cst_2 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%14 : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %15 : tensor<32x64xf32>
  }
}
```

