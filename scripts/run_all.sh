set -ex
export L1_CACHE_SIZE=49152
export L2_CACHE_SZIE=2097152
export L3_CACHE_SIZE=1966080
PROJECT_DIR=`pwd`/../
BUILD_DIR=${PROJECT_DIR}/build
export PYTHONPATH=${PROJECT_DIR}/build/python_packages/gc_mlir_core
export LD_PRELOAD="/home/xiaohui/miniforge3/envs/pytorch/lib/libiomp5.so ${PROJECT_DIR}/build/lib/libGcCpuRuntime.so /home/xiaohui/miniforge3/envs/py11/lib/libjemalloc.so"
export MLIR_RUNNER_UTILS=${PROJECT_DIR}/externals/llvm-project/build/lib/libmlir_runner_utils.so
export MLIR_C_RUNNER_UTILS=${PROJECT_DIR}/externals/llvm-project/build/lib/libmlir_c_runner_utils.so 

cd $BUILD_DIR
# cmake --build . 


echo "thread, dtype, bs, hidden_size, tile, time(ms), GFlops, Correctness, extra"
for tile in 32 64 128
do
for thread in 1 32 56
do
for mode in f32_4dx4d_generic bf16_4dx4d
do

for hidden_size in 4096x4096 4096x11008 11008x4096 4096x32000
do
for bs in 1 16 32 64 512
do
	export OMP_NUM_THREADS=$thread
    M_SIZE=$bs
    N_SIZE=`python -c "print(int('$hidden_size'.split('x')[1]))"`
    K_SIZE=`python -c "print(int('$hidden_size'.split('x')[0]))"`
    TILE_M=`python -c "print(min(int($M_SIZE), int($tile)))"`
    TILE_N=`python -c "print(min(int($N_SIZE), int($tile)))"`
    TILE_K=`python -c "print(min(int($K_SIZE), int($tile)))"`
    PERF_MODE=$mode
    FLOP=`python -c "print(${M_SIZE} * ${K_SIZE} * ${N_SIZE} * 2 / ${thread})"`
	repeat=`python -c "print(1 if $FLOP > 1e12 else 10 if $FLOP > 5e11 else 100)"`
    REPEAT=$repeat

    python ../scripts/generate_single_matmul_mlir.py --M=${M_SIZE} --N=${N_SIZE} --K=${K_SIZE} --tile_m=${TILE_M} --tile_n=${TILE_N} --tile_k=${TILE_K} --mode=${PERF_MODE}  > ${BUILD_DIR}/mlp.mlir
    numactl -N 1 --membind=1 python3 ${PROJECT_DIR}/tools/main.py --type=bench --driver=load_mlir --path=${PROJECT_DIR}/build/mlp.mlir
    exit
    # elapsed_time=`numactl -N 1 --membind=1 python3 ${PROJECT_DIR}/tools/main.py --type=bench --driver=load_mlir --path=${PROJECT_DIR}/build/mlp.mlir 2>log | tail -n 2 | head -n 1 | sed 's/.*execute_cost":.//g'`
    # gflops=`python -c "print(int($bs) * (int)('$hidden_size'.split('x')[0]) * (int)('$hidden_size'.split('x')[1]) * 2 / $elapsed_time / 1e6)"`
	# correctness=`numactl -N 1 --membind=1 python3 ${PROJECT_DIR}/tools/example/simple_test.py --M=${M_SIZE} --K=${K_SIZE} --N=${N_SIZE} --tile_m=${TILE_M} --tile_n=${TILE_N} --tile_k=${TILE_K} --mode=${PERF_MODE} | tail -n 1`
    # echo "$thread,$mode,$bs,$hidden_size,$tile,$elapsed_time,${gflops},${correctness},$1"
done
done

for size in 1024 2048 4096 8192
do
	export OMP_NUM_THREADS=$thread
    M_SIZE=$size
    N_SIZE=$size
    K_SIZE=$size
    TILE_M=`python -c "print(min(int($M_SIZE), int($tile)))"`
    TILE_N=`python -c "print(min(int($N_SIZE), int($tile)))"`
    TILE_K=`python -c "print(min(int($K_SIZE), int($tile)))"`
    PERF_MODE=$mode
    FLOP=`python -c "print(${M_SIZE} * ${K_SIZE} * ${N_SIZE} * 2 / ${thread})"`
	repeat=`python -c "print(1 if $FLOP > 1e12 else 10 if $FLOP > 5e11 else 100)"`
    REPEAT=$repeat

    python ../scripts/generate_single_matmul_mlir.py --M=${M_SIZE} --N=${N_SIZE} --K=${K_SIZE} --tile_m=${TILE_M} --tile_n=${TILE_N} --tile_k=${TILE_K} --mode=${PERF_MODE}  > ${BUILD_DIR}/mlp.mlir

    # elapsed_time=`numactl -N 1 --membind=1 python3 ${PROJECT_DIR}/tools/main.py --type=bench --driver=load_mlir --path=${PROJECT_DIR}/build/mlp.mlir 2>log | tail -n 2 | head -n 1 | sed 's/.*execute_cost":.//g'`
    # gflops=`python -c "print(int($K_SIZE) * (int)($M_SIZE) * (int)($N_SIZE) * 2 / $elapsed_time / 1e6)"`
    # correctness=`numactl -N 1 --membind=1 python3 ${PROJECT_DIR}/tools/example/simple_test.py --M=${M_SIZE} --K=${K_SIZE} --N=${N_SIZE} --tile_m=${TILE_M} --tile_n=${TILE_N} --tile_k=${TILE_K} --mode=${PERF_MODE} | tail -n 1`
	# echo "$thread,$mode,$M_SIZE,${K_SIZE}x${N_SIZE},$tile,$elapsed_time,${gflops},${correctness},$1"
done

done
done
done