#!/bin/bash
cd "$(dirname "$0")"; export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8; PY=/c/Users/Micro/miniconda3/python
for s in 1 2 3; do for a in resonant shuffled; do $PY train.py fit $a $s 3000 3e-3 > logs/${a}_seed$s.txt 2>&1 & done; done
for s in 1 2 3; do
  $PY train.py fit wiener_hammerstein $s 3000 3e-3 > logs/wiener_hammerstein_seed$s.txt 2>&1
  for a in blackbox_small blackbox_large blackbox_canonical; do
    $PY train.py fit $a $s 6000 1e-3 _lr1 > logs/${a}_lr1_seed$s.txt 2>&1
    $PY train.py fit $a $s 6000 3e-3 _lr3 > logs/${a}_lr3_seed$s.txt 2>&1
  done
done
wait; echo ALL_DONE
