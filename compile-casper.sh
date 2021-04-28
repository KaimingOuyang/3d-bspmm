#!/bin/bash

which mpicc

MPIFLAGS="$(mpicc -show | sed 's/^gcc//')"
echo ${MPIFLAGS}
CFLAGS="-fpie -pie -rdynamic -pthread -O2 -DOUTPUT_ALL_PHASES -I$HOME/local/include"
LDFLAGS="-L$HOME/local/lib -Wl,-rpath=$HOME/local/lib -lopenblas -L$HOME/lib/casper-dyn/lib -Wl,-rpath=$HOME/lib/casper-dyn/lib -lcasper"

file=$1
out=$2

gcc ${CFLAGS} -o ${out} ${file} ${LDFLAGS} ${MPIFLAGS}
