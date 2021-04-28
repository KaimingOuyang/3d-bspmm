#!/bin/bash

which mpicc

CFLAGS="-O2 -DOUTPUT_ALL_PHASES"
LDFLAGS="-lopenblas"

file=$1
out=$2

mpicc ${CFLAGS} -o ${out} ${file} ${LDFLAGS}
