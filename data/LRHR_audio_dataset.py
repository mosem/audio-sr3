from torch.utils.data import Dataset
import os
import json
import math

from torch.nn import functional as F

from .audio import Audioset

import logging
logger = logging.getLogger('base')

def match_hr_signal_to_sr_sig(hr_sig, sr_sig):
    hr_len = hr_sig.shape[-1]
    sr_len = sr_sig.shape[-1]
    if hr_len < sr_len:
        hr_sig = F.pad(hr_sig, (0, sr_len - hr_sig.shape[-1]))
    elif hr_len > sr_len:
        hr_sig = hr_sig[..., :sr_len]
    return hr_sig


def compute_output_length(length, depth):
    for i in range(depth - 1):
        length = math.floor((length - 1) / 2 + 1)
    for j in range(depth - 1):
        length = math.floor(length * 2)
    return length


class LRHRAudioDataset(Dataset):

    def __init__(self, json_dir, lr_sr=8000, hr_sr=16000, stride=None, segment=None, pad=True,
                 split='train', data_len=-1, need_LR=False):
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.lr_sr = lr_sr
        self.hr_sr = hr_sr

        lr_stride = stride * lr_sr if stride else None
        hr_stride = stride * hr_sr if stride else None
        lr_length = segment * lr_sr if segment else None
        hr_length = segment * hr_sr if segment else None


        sr_json = os.path.join(json_dir, 'sr.json')
        hr_json = os.path.join(json_dir, 'hr.json')

        with open(sr_json, 'r') as f:
            sr = json.load(f)
        with open(hr_json, 'r') as f:
            hr = json.load(f)

        sr.sort()
        hr.sort()

        self.sr_set = Audioset(sr, sample_rate=hr_sr, length=hr_length, stride=hr_stride, pad=pad, channels=1)
        self.hr_set = Audioset(hr, sample_rate=hr_sr, length=hr_length, stride=hr_stride, pad=pad, channels=1)
        assert len(self.hr_set) == len(self.sr_set)

        self.dataset_len = len(self.hr_set)

        if self.need_LR:
            lr_json = os.path.join(json_dir, 'lr.json')
            with open(lr_json, 'r') as f:
                lr = json.load(f)
            self.lr_set = Audioset(lr, sample_rate=lr_sr, length=lr_length, stride=lr_stride, pad=pad, channels=1)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        hr_sig, hr_filename = self.hr_set[index]
        sr_sig, sr_filename = self.sr_set[index]

        hr_sig = match_hr_signal_to_sr_sig(hr_sig, sr_sig)

        if self.split == 'val':
            sig_len = compute_output_length(hr_sig.shape[-1], 5)
            hr_sig = F.pad(hr_sig, (0, sig_len - hr_sig.shape[-1]))
            sr_sig = F.pad(sr_sig, (0, sig_len - sr_sig.shape[-1]))

        if self.need_LR:
            lr_sig, lr_filename = self.lr_set[index]
            # augment?
            return {'LR': lr_sig, 'HR': hr_sig, 'SR': sr_sig, 'Filename': hr_filename}
        else:
            # augment?
            return {'HR': hr_sig, 'SR': sr_sig, 'Filename': hr_filename}