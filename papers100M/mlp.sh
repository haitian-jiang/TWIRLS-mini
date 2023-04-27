#!/bin/bash
for alpha in 1.0 5.0
do
    for lmbd in 0.01 0.1 1.0 10.0 100.0
    do
        python mlp.py --dataset=8-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.2 --device=0 &
        python mlp.py --dataset=8-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.5 --device=1 &
        python mlp.py --dataset=8-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.2 --device=2 &
        python mlp.py --dataset=8-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.5 --device=3 &
        python mlp.py --dataset=16-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.2 --device=4 &
        python mlp.py --dataset=16-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.5 --device=5 &
        python mlp.py --dataset=16-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.2 --device=6 &
        python mlp.py --dataset=16-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.5 --device=7 &
        wait
    done
done

K=32 alpha=0.1
for lmbd in 1.0 10.0 100.0
do
    python mlp.py --dataset=${K}-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.2 --device=0 &
    python mlp.py --dataset=${K}-${alpha}-${lmbd} --lr=1e-2 --wd=0 --dropout=0.5 --device=1 &
    python mlp.py --dataset=${K}-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.2 --device=2 &
    python mlp.py --dataset=${K}-${alpha}-${lmbd} --lr=1e-2 --wd=5e-4 --dropout=0.5 --device=3 &
    wait
done