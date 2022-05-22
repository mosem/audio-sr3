SPEAKER_DIR=p225

HR_DIR=hr_16000
LR_DIR=lr_8000
SR_DIR=sr_8000_16000

hr_speaker_dir_path=$(realpath ./$(HR_DIR))
lr_speaker_dir_path=$(realpath ./$(LR_DIR))
sr_speaker_dir_path=$(realpath ./$(SR_DIR))

all:
	make train-files-$(HR_DIR).txt val-files-$(HR_DIR).txt
	make train-files-$(LR_DIR).txt val-files-$(LR_DIR).txt
	make train-files-$(SR_DIR).txt val-files-$(SR_DIR).txt

train-files-$(HR_DIR).txt:
	find $(hr_speaker_dir_path) | grep -F '.wav' | sort | head -n 223 > $@

val-files-$(HR_DIR).txt:
	find $(hr_speaker_dir_path)| grep -F '.wav' | sort | tail -n 8 > $@

train-files-$(LR_DIR).txt:
	find $(lr_speaker_dir_path) | grep -F '.wav' | sort | head -n 223 > $@

val-files-$(LR_DIR).txt:
	find $(lr_speaker_dir_path)| grep -F '.wav' | sort | tail -n 8 > $@

train-files-$(SR_DIR).txt:
	find $(sr_speaker_dir_path) | grep -F '.wav' | sort | head -n 223 > $@

val-files-$(SR_DIR).txt:
	find $(sr_speaker_dir_path)| grep -F '.wav' | sort | tail -n 8 > $@
