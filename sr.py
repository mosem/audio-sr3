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
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

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
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    logger.info(f'train set length: {len(train_set)}')
    logger.info(f'train loader length: {len(train_loader)}')
    logger.info(f'val set length: {len(val_set)}')
    logger.info(f'val loader length: {len(val_loader)}')

    lr_sr = opt['datasets']['val']['lr_sr']
    hr_sr = opt['datasets']['val']['hr_sr']

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    n_params = sum(p.numel() for p in diffusion.netG.parameters() if p.requires_grad)
    mb = n_params * 4 / 2 ** 20
    logger.info(f"{opt['model']['which_model_G']}: parameters: {n_params}, size: {mb} MB")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4} '.format(k, float(v))
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_sisnr = 0.0
                    avg_lsd = 0.0
                    avg_visqol = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        sr_audio = Metrics.tensor2audio(visuals['SR'])  # uint8
                        hr_audio = Metrics.tensor2audio(visuals['HR'])  # uint8
                        lr_audio = Metrics.tensor2audio(visuals['LR'])  # uint8
                        fake_audio = Metrics.tensor2audio(visuals['INF'])  # uint8



                        # generation
                        Metrics.save_audio(
                            hr_audio, '{}/{}_{}_hr.wav'.format(result_path, current_step, idx), hr_sr)
                        Metrics.save_audio(
                            sr_audio, '{}/{}_{}_sr.wav'.format(result_path, current_step, idx), hr_sr)
                        Metrics.save_audio(
                            lr_audio, '{}/{}_{}_lr.wav'.format(result_path, current_step, idx), lr_sr)
                        Metrics.save_audio(
                            fake_audio, '{}/{}_{}_inf.wav'.format(result_path, current_step, idx), hr_sr)
                        #TODO: fix this
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_audio, sr_audio, hr_audio), axis=1), [2, 0, 1]),
                        #     idx)

                        filename = f'validation_{idx}'

                        sisnr = Metrics.calculate_sisnr(sr_audio, hr_audio)
                        lsd = Metrics.calculate_lsd(sr_audio, hr_audio)
                        visqol = Metrics.calculate_visqol(sr_audio.numpy(), hr_audio.numpy(), filename, hr_sr)
                        avg_sisnr += sisnr
                        avg_lsd += lsd
                        avg_visqol += visqol

                        if wandb_logger:
                            wandb_logger.log_audio(filename, sr_audio, hr_audio, fake_audio, sisnr, lsd, visqol,
                                                   current_step, hr_sr)

                    avg_sisnr = avg_sisnr / idx
                    avg_lsd = avg_lsd / idx
                    avg_visqol = avg_visqol / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # SISNR: {:.4}, LSD: {:.4}, VISQOL: {:.4}'.format(
                        avg_sisnr, avg_lsd, avg_visqol))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> sisnr: {:.4}, lsd: {:.4}, visqol: {:.4}'.format(
                        current_epoch, current_step, avg_sisnr, avg_lsd, avg_visqol))
                    # tensorboard logger
                    tb_logger.add_scalar('sisnr', avg_sisnr, current_step)

                    if wandb_logger:

                        wandb_logger.log_metrics({
                            'validation/val_sisnr': avg_sisnr,
                            'validation/val_lsd': avg_lsd,
                            'validation/val_visqol': avg_visqol,
                            'validation/val_step': val_step
                        })
                        val_step += 1


                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_sisnr = 0.0
        avg_lsd = 0.0
        avg_visqol = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_audio = Metrics.tensor2audio(visuals['HR'])  # uint8
            lr_audio = Metrics.tensor2audio(visuals['LR'])  # uint8
            fake_audio = Metrics.tensor2audio(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_audio = visuals['SR']  # uint8
                sample_num = sr_audio.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_audio(
                        Metrics.tensor2audio(sr_audio[iter]), '{}/{}_{}_sr_{}.wav'.format(result_path, current_step, idx, iter), hr_sr)
            else:
                # grid img
                sr_audio = Metrics.tensor2audio(visuals['SR'])  # uint8
                Metrics.save_audio(
                    sr_audio, '{}/{}_{}_sr_process.wav'.format(result_path, current_step, idx), hr_sr)
                Metrics.save_audio(
                    Metrics.tensor2audio(visuals['SR'][-1]), '{}/{}_{}_sr.wav'.format(result_path, current_step, idx), hr_sr)

            Metrics.save_audio(
                hr_audio, '{}/{}_{}_hr.wav'.format(result_path, current_step, idx), hr_sr)
            Metrics.save_audio(
                lr_audio, '{}/{}_{}_lr.wav'.format(result_path, current_step, idx), lr_sr)
            Metrics.save_audio(
                fake_audio, '{}/{}_{}_inf.wav'.format(result_path, current_step, idx),hr_sr)

            filename = f"{current_step}_{idx}"
            # generation
            eval_sisnr = Metrics.calculate_sisnr(Metrics.tensor2audio(visuals['SR'][-1]), hr_audio)
            eval_lsd = Metrics.calculate_lsd(Metrics.tensor2audio(visuals['SR'][-1]), hr_audio)
            eval_visqol = Metrics.calculate_visqol(Metrics.tensor2audio(visuals['SR'][-1]).numpy(), hr_audio.numpy(), filename, hr_sr)

            avg_sisnr += eval_sisnr
            avg_lsd += eval_lsd
            avg_visqol += eval_visqol

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(filename, fake_audio, Metrics.tensor2audio(visuals['SR'][-1]), hr_audio, eval_sisnr,
                                           eval_lsd, eval_visqol)

        avg_sisnr = avg_sisnr / idx
        avg_lsd = avg_lsd / idx
        avg_visqol = avg_visqol / idx

        # log
        logger.info('# Validation # SISNR: {:.4}'.format(avg_sisnr))
        logger.info('# Validation # LSD: {:.4}'.format(avg_lsd))
        logger.info('# Validation # VISQOL: {:.4}'.format(avg_visqol))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> sisnr: {:.4}, lsd：{:.4}, visqol：{:.4}'.format(
            current_epoch, current_step, avg_sisnr, avg_lsd, avg_visqol))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'SISNR': float(avg_sisnr),
                'LSD': float(avg_lsd),
                'VISQOL': float(avg_visqol)
            })
