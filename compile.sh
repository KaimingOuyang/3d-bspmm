#!/bin/bash

which mpicc

CFLAGS="-fpie -pie -rdynamic -pthread -O2 -DOUTPUT_ALL_PHASES -I$HOME/local/include"
LDFLAGS="-L$HOME/local/lib -Wl,-rpath=$HOME/local/lib -lopenblas"

file=$1
out=$2

mpicc ${CFLAGS} -o ${out} ${file} ${LDFLAGS}
