#!/bin/bash -e

if [ -z "$PATH_TO_JEMALLOC" ]; then
    echo "PATH_TO_JEMALLOC not set."
    exit 1
fi

export LD_PRELOAD=${LD_PRELOAD}:${PATH_TO_JEMALLOC}

for arg in "$@"; do
    case $arg in
        --bench)   
            MODE=P 
            ;;
        --tune)
            MODE=T
            ;;
        *) 
            echo Unsupported option: $arg
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Mode not set."
    exit 1
fi

if [ -z "$NUM_THREADS" ]; then
    echo "NUM_THREADS not set."
    exit 1
fi

export OMP_NUM_THREADS=${NUM_THREADS}
export START_NODE=0
export END_NODE=$(($OMP_NUM_THREADS-1))

if [ "$MODE" = "P" ]; then
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=16x512x256x128 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --has_bias=1x1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=32 --hidden_size_list=4096x4096x11008x4096 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 500
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=4096x4096x11008x4096 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 500 
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=16x512x256x128 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --dtype=bf16 --has_bias=1x1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=32 --hidden_size_list=4096x4096x11008x4096 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 2000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=4096x4096x11008x4096 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 2000
else
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=16x512x256x128 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --has_bias=1x1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=32 --hidden_size_list=4096x4096x11008x4096 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 500
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=4096x4096x11008x4096 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 500 
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=16x512x256x128 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --dtype=bf16 --has_bias=1x1x1x1 --act_type=relu --warm_up 500 --repeat 5000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=32 --hidden_size_list=4096x4096x11008x4096 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 2000
    numactl -C $START_NODE-$END_NODE -m 0 python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=4096x4096x11008x4096 --dtype=bf16 --has_bias=1x1x1 --act_type=relu --warm_up 500 --repeat 2000
fi
