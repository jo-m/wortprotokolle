#!/bin/bash

run="$1"
[ -z "$1" ] && run='train.py'

if [ "$run" = 'repl' ]; then
    run=''
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
