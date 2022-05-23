import argparse

import torchaudio

import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.wav'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.wav'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_sisnr = 0.0
    avg_lsd = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_hr")[0]
        assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
            ridx, fidx)

        hr_audio, _ = torchaudio.load(rname)
        sr_audio, _ = torchaudio.load(fname)
        sisnr = Metrics.calculate_sisnr(Metrics.tensor2audio(sr_audio), hr_audio)
        lsd = Metrics.calculate_lsd(Metrics.tensor2audio(sr_audio), hr_audio)
        avg_sisnr += sisnr
        avg_lsd += lsd

        if idx % 20 == 0:
            print('Audio:{}, SISNR:{:.4f}, LSD:{:.4f}'.format(idx, sisnr, lsd))

    avg_sisnr = avg_sisnr / idx
    avg_lsd = avg_lsd / idx

    # log
    print('# Validation # SISNR: {:.4e}'.format(avg_sisnr))
    print('# Validation # LSD: {:.4e}'.format(avg_lsd))
