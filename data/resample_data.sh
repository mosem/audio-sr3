#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

sr=$1

find /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_wav/* -type d -name "*" -prune | xargs -n 1 readlink -f | xargs -n 1 -P 10 -0 -t -r -I {} sox {} -r ${sr} ${sr}Hz/{}
