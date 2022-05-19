#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

cd /cs/labs/adiyoss/moshemandel/sr3-audio/code

python sr.py -p train -c config/sr_sr3_16_128.json