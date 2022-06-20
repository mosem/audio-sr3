#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

cd /cs/labs/adiyoss/moshemandel/sr3-audio/code

python infer_and_eval.py -c config/vctk_8_16/single/hdemucs/sr_hdemucs_single_8000_16000_dummy_infer.json
                          -enable_wandb \
                          -log_eval \