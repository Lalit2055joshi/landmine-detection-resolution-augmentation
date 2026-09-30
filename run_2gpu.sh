#!/usr/bin/env bash
# Same pipeline on a server with 2 GPUs: YOLOv8-S on GPU 0, YOLOv10-S on GPU 1.
# Roughly halves the wall-clock time.
#
#   nohup bash run_2gpu.sh > logs/run_2gpu.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python check_data.py

python train.py --only v8s_  --device 0 > logs/train_v8s.log  2>&1 &
P0=$!
python train.py --only v10s_ --device 1 > logs/train_v10s.log 2>&1 &
P1=$!
wait $P0; wait $P1   # set -e stops here if either run failed

python evaluate.py --only v8s_  --device 0 > logs/eval_v8s.log  2>&1 &
P0=$!
python evaluate.py --only v10s_ --device 1 > logs/eval_v10s.log 2>&1 &
P1=$!
wait $P0; wait $P1   # set -e stops here if either run failed

python collect.py
echo "ALL DONE — see results/tables.md"
