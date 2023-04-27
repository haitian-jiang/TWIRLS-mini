#!/bin/bash
for K in 8 16
do
    for alpha in 1.0 5.0
    do
        for lmbd in 0.01 0.1 1.0 10.0 100.0
        do
            python prop.py --K=$K --alpha=$alpha --lmbd=$lmbd
        done
    done
done

K=32 alpha=0.1
for lmbd in 1.0 10.0 100.0
do
    python prop.py --K=$K --alpha=$alpha --lmbd=$lmbd
done