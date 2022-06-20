#!/bin/bash

. /cs/labs/adiyoss/moshemandel/sr3-audio/venv-sr3/bin/activate

cd /cs/labs/adiyoss/moshemandel/sr3-audio/code


# clean train sets

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_trainset_56spk_4k \
	--target_sr 4000 \

# noisy train sets

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_trainset_56spk_4k \
	--target_sr 4000 \


# noisy test sets

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/noisy_testset_4k \
	--target_sr 4000 \

# clean test sets

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_24k \
	--target_sr 24000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_16k \
	--target_sr 16000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_8k \
	--target_sr 8000 \

python3 data/prep_audio_data.py  \
	--data_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_wav \
	--out_dir /cs/labs/adiyoss/moshemandel/data/valentini/clean_testset_4k \
	--target_sr 4000 \