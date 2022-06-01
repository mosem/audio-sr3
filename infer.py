import torch
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
    parser.add_argument('-log_infer', action='store_true')
    
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
            logger.info(f'sampler indices: {sampler.indices_sorted_by_len}')
            # val_loader = Data.create_dataloader(
            #     val_set, dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase,sampler,collate_fn)
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
    n_processed_files = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        filenames = val_data['filename']

        filenames_exist = [os.path.isfile(os.path.join(result_path, filename + '_hr.wav')) for filename in filenames]
        if all(filenames_exist):
            logger.info(f'{idx}/{len(val_loader)}) all files already exists: {",".join(filenames)}')
            continue

        logger.info(f'{idx}/{len(val_loader)}) Inferring {",".join(filenames)}.')
        n_processed_files += 1

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)





        for i,filename in enumerate(filenames):
            hr_audio = Metrics.tensor2audio(visuals['HR'][i])
            fake_audio = Metrics.tensor2audio(visuals['INF'][i])
            sr_audio = visuals['SR']

            signal_length = val_data['length'][i]

            hr_audio = hr_audio[:,:signal_length]
            fake_audio = fake_audio[:,:signal_length]
            sr_audio = sr_audio[:,:,:signal_length]


            sr_img_mode = 'grid'
            if sr_img_mode == 'grid':
                sample_num = sr_audio.shape[1]
                for sample_idx in range(0, sample_num - 1):
                    Metrics.save_audio(
                        Metrics.tensor2audio(sr_audio[i, sample_idx:sample_idx+1, :]),
                        '{}/{}_pr_process_{}.wav'.format(result_path, filename, sample_idx), hr_sr)

            Metrics.save_audio(
                Metrics.tensor2audio(sr_audio[i,-1:,:]),
                '{}/{}_pr.wav'.format(result_path, filename), hr_sr)

            Metrics.save_audio(
                hr_audio, '{}/{}_hr.wav'.format(result_path, filename), hr_sr)
            Metrics.save_audio(
                fake_audio, '{}/{}_lr.wav'.format(result_path, filename), hr_sr)

            if wandb_logger and opt['log_infer']:
                wandb_logger.log_eval_data(filename, fake_audio, Metrics.tensor2audio(visuals['SR'][-1]), hr_audio, hr_sr)

    logger.info(f'Done. Processed {n_processed_files}/{len(val_loader)} files.')

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
