#!/bin/bash

run="$1"
[ -z "$1" ] && run='train.py'

if [ "$run" = 'repl' ]; then
    run=''
fi

# enable GPU support
if [ `uname` = Darwin ] && [ -d '/Developer/NVIDIA/CUDA-7.0' ]; then
    export THEANO_FLAGS='cuda.root=/Developer/NVIDIA/CUDA-7.0,device=gpu,floatX=float32,force_device=True'
fi

if [ -x "$HOME/anaconda/bin/python" ]; then
    "$HOME/anaconda/bin/python" $run
else
    python $run
fi

if [ $? -eq 0 ]; then
    say 'run succeeded'
else
    say 'run failed'
fi
