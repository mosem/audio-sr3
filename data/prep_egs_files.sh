#!/bin/bash

# usage example: bash prep_egs_files.sh 16 24

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate
cd /cs/labs/adiyoss/moshemandel/sr3-audio/code/data

lr=$1
hr=$2
mode=${3:-'single-speaker'}
n_samples_limit=${4:--1}
out_dir=$lr-$hr

if [[ $n_samples_limit -gt 0 ]]
then
  out_dir+="(${n_samples_limit})"
fi

echo "saving to $mode/${out_dir}"

tr_out=../egs/vctk/$mode/$out_dir/tr
val_out=../egs/vctk/$mode/$out_dir/val

lr_train_files=../dataset-audio/$mode/train-files-${lr}.txt
sr_train_files=../dataset-audio/$mode/train-files-${lr}_${hr}.txt
hr_train_files=../dataset-audio/$mode/train-files-${hr}.txt

lr_val_files=../dataset-audio/$mode/val-files-${lr}.txt
sr_val_files=../dataset-audio/$mode/val-files-${lr}_${hr}.txt
hr_val_files=../dataset-audio/$mode/val-files-${hr}.txt

mkdir -p $tr_out
mkdir -p $val_out

python -m prep_egs_files $lr_train_files $n_samples_limit > $tr_out/lr.json
python -m prep_egs_files $sr_train_files $n_samples_limit > $tr_out/sr.json
python -m prep_egs_files $hr_train_files $n_samples_limit > $tr_out/hr.json

python -m prep_egs_files $lr_val_files $n_samples_limit > $val_out/lr.json
python -m prep_egs_files $sr_val_files $n_samples_limit > $val_out/sr.json
python -m prep_egs_files $hr_val_files $n_samples_limit > $val_out/hr.json