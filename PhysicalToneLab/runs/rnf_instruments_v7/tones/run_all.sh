#!/bin/bash
cd "$(dirname "$0")"; export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8; PY=/c/Users/Micro/miniconda3/python; mkdir -p logs
run() { $PY tones.py fit $1 $2 $3 $4 > logs/$1_$2_$3.txt 2>&1; }
(for t in BluesDriver TubeScreamer RAT; do run $t resonant 1 1500; done) &
(for t in BluesDriver TubeScreamer RAT; do run $t resonant 2 1500; done) &
(for t in BluesDriver TubeScreamer RAT; do run $t shuffled 1 1500; done) &
(for t in BluesDriver TubeScreamer RAT; do for s in 1 2; do run $t wiener_hammerstein $s 3000; run $t blackbox_large $s 6000; done; done) &
($PY cond.py fit 1 3000 > logs/cond_1.txt 2>&1) &
wait; $PY tones.py test > logs/test.txt 2>&1; $PY cond.py test > logs/test_cond.txt 2>&1; echo ALL_DONE
