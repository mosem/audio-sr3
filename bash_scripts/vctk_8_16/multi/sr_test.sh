#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

cd /cs/labs/adiyoss/moshemandel/sr3-audio/code

python sr.py -p val \
             -c config/vctk_8_16/multi/sr_sr3_multi_8000_16000_infer.json \
             -enable_wandb \
             -log_eval \
