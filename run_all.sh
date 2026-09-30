#!/usr/bin/env bash
# Full pipeline on ONE GPU: check data -> train 8 runs -> evaluate -> tables.
# Safe to re-run: finished runs are skipped and interrupted runs resume.
#
#   nohup bash run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python check_data.py
python train.py
python evaluate.py
python collect.py
echo "ALL DONE — see results/tables.md"
