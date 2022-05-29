#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

cd /cs/labs/adiyoss/moshemandel/sr3-audio/code

python eval.py -p /cs/labs/adiyoss/moshemandel/sr3-audio/code/experiments/sr_vctk_220524_084426/results
