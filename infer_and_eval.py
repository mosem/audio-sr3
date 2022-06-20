import torch
import torchaudio

import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

from data.util import SequentialBinSampler, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--sr', type=int, default=16000)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            sampler = SequentialBinSampler(val_set.get_file_lengths())
            # logger.info(f'sampler indices: {sampler.indices_sorted_by_len}')
            # val_loader = Data.create_dataloader(
            #    val_set, dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, sampler, collate_fn)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    lr_sr = opt['datasets']['val']['lr_sr']
    hr_sr = opt['datasets']['val']['hr_sr']

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    n_processed_batches = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    avg_pesq = 0.0
    avg_stoi = 0.0
    avg_sisnr = 0.0
    avg_lsd = 0.0
    avg_visqol = 0.0
    n_files_evaluated = 0

    for _, val_data in enumerate(val_loader):
        idx += 1
        filenames = val_data['filename']

        filenames_exist = [os.path.isfile(os.path.join(result_path, filename + '_hr.wav')) for filename in filenames]
        if all(filenames_exist):
            logger.info(f'{idx}/{len(val_loader)}) all files already exists: {",".join(filenames)}')
            for filename in filenames:
                hr_name = os.path.join(result_path, filename + '_hr.wav')
                sr_name = os.path.join(result_path, filename + '_pr.wav')
                hr_audio, _ = torchaudio.load(hr_name)
                sr_audio, _ = torchaudio.load(sr_name)
                hr_audio = Metrics.tensor2audio(hr_audio)
                sr_audio = Metrics.tensor2audio(sr_audio)

                pesq = Metrics.calculate_pesq(sr_audio.numpy(), hr_audio.numpy(), hr_sr)
                stoi = Metrics.calculate_stoi(sr_audio.numpy(), hr_audio.numpy(), hr_sr)
                sisnr = Metrics.calculate_sisnr(sr_audio, hr_audio)
                lsd = Metrics.calculate_lsd(sr_audio, hr_audio)
                visqol = Metrics.calculate_visqol(sr_audio.numpy(), hr_audio.numpy(), filename, args.sr)
                avg_pesq += pesq
                avg_stoi += stoi
                avg_sisnr += sisnr
                avg_lsd += lsd
                avg_visqol += visqol
                n_files_evaluated += 1

                lr_name = os.path.join(result_path, filename + '_lr.wav')
                lr_audio, _ = torchaudio.load(lr_name)

                if wandb_logger and opt['log_eval']:
                    wandb_logger.log_eval_data(filename, lr_audio, sr_audio, hr_audio,
                                               hr_sr,
                                               pesq, stoi, sisnr, lsd, visqol)

            continue

        logger.info(f'{idx}/{len(val_loader)}) Inferring {",".join(filenames)}.')
        n_processed_batches += 1

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        for i, filename in enumerate(filenames):
            hr_audio = Metrics.tensor2audio(visuals['HR'][i])
            lr_audio = Metrics.tensor2audio(visuals['INF'][i])
            sr_audio = visuals['SR']

            signal_length = val_data['length'][i]

            hr_audio = hr_audio[:, :signal_length]
            lr_audio = lr_audio[:, :signal_length]
            sr_audio = sr_audio[:, :, :signal_length]

            sr_img_mode = 'grid'
            if sr_img_mode == 'grid':
                sample_num = sr_audio.shape[1]
                for sample_idx in range(0, sample_num - 1):
                    Metrics.save_audio(
                        Metrics.tensor2audio(sr_audio[i, sample_idx:sample_idx + 1, :]),
                        '{}/{}_pr_process_{}.wav'.format(result_path, filename, sample_idx), hr_sr)

            Metrics.save_audio(
                Metrics.tensor2audio(sr_audio[i, -1:, :]),
                '{}/{}_pr.wav'.format(result_path, filename), hr_sr)

            Metrics.save_audio(
                hr_audio, '{}/{}_hr.wav'.format(result_path, filename), hr_sr)
            Metrics.save_audio(
                lr_audio, '{}/{}_lr.wav'.format(result_path, filename), hr_sr)

            pesq = Metrics.calculate_pesq(sr_audio[i, -1:, :].numpy(), hr_audio.numpy(), hr_sr)
            stoi = Metrics.calculate_stoi(sr_audio[i, -1:, :].numpy(), hr_audio.numpy(), hr_sr)
            sisnr = Metrics.calculate_sisnr(sr_audio[i, -1:, :], hr_audio)
            lsd = Metrics.calculate_lsd(sr_audio[i, -1:, :], hr_audio)
            visqol = Metrics.calculate_visqol(sr_audio[i, -1:, :].numpy(), hr_audio.numpy(), filename, args.sr)
            avg_pesq += pesq
            avg_stoi += stoi
            avg_sisnr += sisnr
            avg_lsd += lsd
            avg_visqol += visqol
            n_files_evaluated += 1

            if wandb_logger and opt['log_infer']:
                wandb_logger.log_eval_data(filename, lr_audio, Metrics.tensor2audio(sr_audio[i, -1:, :]), hr_audio,
                                           hr_sr,
                                           pesq, stoi, sisnr, lsd, visqol)

    logger.info(f'Done. Processed {n_processed_batches}/{len(val_loader)} batches.')

    avg_pesq = avg_pesq / n_files_evaluated
    avg_stoi = avg_stoi / n_files_evaluated
    avg_sisnr = avg_sisnr / n_files_evaluated
    avg_lsd = avg_lsd / n_files_evaluated
    avg_visqol = avg_visqol / n_files_evaluated

    logger.info('# Validation # PESQ: {:.4e}'.format(avg_pesq))
    logger.info('# Validation # STOI: {:.4e}'.format(avg_stoi))
    logger.info('# Validation # SISNR: {:.4e}'.format(avg_sisnr))
    logger.info('# Validation # LSD: {:.4e}'.format(avg_lsd))
    logger.info('# Validation # VISQOL: {:.4e}'.format(avg_visqol))

    if wandb_logger and opt['log_eval']:
        wandb_logger.log_metrics({
            'PESQ': avg_pesq,
            'STOI': avg_stoi,
            'SISNR': avg_sisnr,
            'LSD': avg_lsd,
            'VISQOL': avg_visqol
        })
        wandb_logger.log_eval_table(commit=True)
