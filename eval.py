import argparse
import os

import torchaudio

import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')

    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.wav'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.wav'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_sisnr = 0.0
    avg_lsd = 0.0
    avg_visqol = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = fname.rsplit("_sr")[0]
        assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
            ridx, fidx)

        hr_audio, _ = torchaudio.load(rname)
        sr_audio, _ = torchaudio.load(fname)
        sr_audio = Metrics.tensor2audio(sr_audio)
        sisnr = Metrics.calculate_sisnr(sr_audio, hr_audio)
        lsd = Metrics.calculate_lsd(sr_audio, hr_audio)
        visqol = Metrics.calculate_visqol(sr_audio.numpy(), hr_audio.numpy(), os.path.basename(ridx), args.sr)
        avg_sisnr += sisnr
        avg_lsd += lsd
        avg_visqol += visqol

        if idx % 20 == 0:
            print('Audio:{}, SISNR:{:.4f}, LSD:{:.4f}, VISQOL: {:.4f}'.format(idx, sisnr, lsd, visqol))

    avg_sisnr = avg_sisnr / idx
    avg_lsd = avg_lsd / idx
    avg_visqol = avg_visqol / idx

    # log
    print('# Validation # SISNR: {:.4e}'.format(avg_sisnr))
    print('# Validation # LSD: {:.4e}'.format(avg_lsd))
    print('# Validation # VISQOL: {:.4e}'.format(avg_visqol))
