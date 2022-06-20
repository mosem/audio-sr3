import os
from torchaudio.transforms import Spectrogram
import numpy as np
import cv2

def scale_minmax(X, min=0.0, max=1.0):
    isnan = np.isnan(X).any()
    isinf = np.isinf(X).any()
    if isinf:
        X[X == np.inf] = 1e9
        X[X == -np.inf] = 1e-9
    if isnan:
        X[X == np.nan] = 1e-9
    # logger.info(f'isnan: {isnan}, isinf: {isinf}, max: {X.max()}, min: {X.min()}')

    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def convert_spectrogram_to_heatmap(spectrogram):
    spectrogram += 1e-9
    spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8).squeeze()
    spectrogram = np.flip(spectrogram, axis=0)
    spectrogram = 255 - spectrogram
    # spectrogram = (255 * (spectrogram - np.min(spectrogram)) / np.ptp(spectrogram)).astype(np.uint8).squeeze()[::-1,:]
    heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        WANDB_ENTITY = 'huji-dl-audio-lab'

        self._wandb = wandb

        self.files_initially_logged = set()

        # Initialize a W&B run
        if self._wandb.run is None:
            if 'id' in opt['wandb']:
                self._wandb.init(
                    id=opt['wandb']['id'],
                    resume='must',
                    project=opt['wandb']['project'],
                    entity=WANDB_ENTITY,
                    config=opt,
                    group=opt['wandb']['group'],
                    name=opt['wandb']['name'],
                    dir='./experiments',
                )
            else:
                self._wandb.init(
                    project=opt['wandb']['project'],
                    entity=WANDB_ENTITY,
                    config=opt,
                    group=opt['wandb']['group'],
                    name=opt['wandb']['name'],
                    dir='./experiments'
                )

        self.config = self._wandb.config

        if self.config.get('log_eval', None):
            self.eval_table = self._wandb.Table(columns=['lr_audio',
                                                         'pr_audio',
                                                         'hr_audio',
                                                         'lr_spec',
                                                         'pr_spec',
                                                         'hr_spec',
                                                         'pesq',
                                                         'stoi',
                                                         'sisnr',
                                                         'lsd',
                                                         'visqol'])
        else:
            self.eval_table = None

        if self.config.get('log_infer', None):
            self.infer_table = self._wandb.Table(columns=['lr_audio',
                                                          'pr_audio',
                                                          'hr_audio',
                                                          'lr_spec',
                                                          'pr_spec',
                                                          'hr_spec'])
        else:
            self.infer_table = None

    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        self._wandb.log(metrics, commit=commit)

    def log_audio(self, filename, pr_signal, hr_signal, lr_signal, sisnr, lsd, visqol, epoch, sr):
        spectrogram_transform = Spectrogram()
        enhanced_spectrogram = spectrogram_transform(pr_signal).log2()[0, :, :].numpy()
        enhanced_spectrogram_wandb_image = self._wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram),
                                                       caption=f'{filename}_pr')
        enhanced_wandb_audio = self._wandb.Audio(pr_signal.squeeze().numpy(), sample_rate=sr, caption=filename)

        wandb_dict = {f'test samples/{filename}/lsd': lsd,
                      f'test samples/{filename}/sisnr': sisnr,
                      f'test samples/{filename}/visqol': visqol,
                      f'test samples/{filename}/spectrogram': enhanced_spectrogram_wandb_image,
                      f'test samples/{filename}/audio': enhanced_wandb_audio}

        if filename not in self.files_initially_logged:
            self.files_initially_logged.add(filename)
            hr_name = f'{filename}_hr'
            hr_enhanced_spectrogram = spectrogram_transform(hr_signal).log2()[0, :, :].numpy()
            hr_enhanced_spectrogram_wandb_image = self._wandb.Image(convert_spectrogram_to_heatmap(hr_enhanced_spectrogram),
                                                              caption=hr_name)
            hr_enhanced_wandb_audio = self._wandb.Audio(hr_signal.squeeze().numpy(), sample_rate=sr, caption=hr_name)
            wandb_dict.update({f'test samples/{filename}/{hr_name}_spectrogram': hr_enhanced_spectrogram_wandb_image,
                               f'test samples/{filename}/{hr_name}_audio': hr_enhanced_wandb_audio})

            lr_name = f'{filename}_lr'
            lr_enhanced_spectrogram = spectrogram_transform(lr_signal).log2()[0, :, :].numpy()
            lr_enhanced_spectrogram_wandb_image = self._wandb.Image(
                convert_spectrogram_to_heatmap(lr_enhanced_spectrogram),
                caption=lr_name)
            lr_enhanced_wandb_audio = self._wandb.Audio(lr_signal.squeeze().numpy(), sample_rate=sr, caption=lr_name)
            wandb_dict.update({f'test samples/{filename}/{lr_name}_spectrogram': lr_enhanced_spectrogram_wandb_image,
                               f'test samples/{filename}/{lr_name}_audio': lr_enhanced_wandb_audio})

        self._wandb.log(wandb_dict,
                  step=epoch)

    def log_image(self, key_name, image_array):
        """
        Log image array onto W&B.

        key_name: name of the key 
        image_array: numpy array of image.
        """
        self._wandb.log({key_name: self._wandb.Image(image_array)})

    def log_images(self, key_name, list_images):
        """
        Log list of image array onto W&B

        key_name: name of the key 
        list_images: list of numpy image arrays
        """
        self._wandb.log({key_name: [self._wandb.Image(img) for img in list_images]})

    def log_checkpoint(self, current_epoch, current_step):
        """
        Log the model checkpoint as W&B artifacts

        current_epoch: the current epoch 
        current_step: the current batch step
        """
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        gen_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_gen.pth'.format(current_step, current_epoch))
        opt_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_opt.pth'.format(current_step, current_epoch))

        model_artifact.add_file(gen_path)
        model_artifact.add_file(opt_path)
        self._wandb.log_artifact(model_artifact, aliases=["latest"])

    def log_eval_data(self, filename, fake_audio, sr_audio, hr_audio, sr, pesq=None, stoi=None, sisnr=None, lsd=None, visqol=None):
        """
        Add data row-wise to the initialized table.
        """
        SPECTOGRAM_EPSILON = 1e-13
        spectrogram_transform = Spectrogram(n_fft=800)

        hr_audio_spectrogram = spectrogram_transform(hr_audio).log2()[0, :, :].numpy()
        lr_spectrogram = (SPECTOGRAM_EPSILON + spectrogram_transform(fake_audio)).log2()[0, :, :].numpy()
        sr_audio_spectrogram = spectrogram_transform(sr_audio).log2()[0, :, :].numpy()
        hr_audio_wandb_spec = self._wandb.Image(convert_spectrogram_to_heatmap(hr_audio_spectrogram))
        lr_wandb_spec = self._wandb.Image(convert_spectrogram_to_heatmap(lr_spectrogram))
        sr_audio_wandb_spec = self._wandb.Image(convert_spectrogram_to_heatmap(sr_audio_spectrogram))


        hr_wandb_audio = self._wandb.Audio(hr_audio.squeeze().numpy(), sample_rate=sr, caption=filename + '_hr')
        lr_wandb_audio = self._wandb.Audio(fake_audio.squeeze().numpy(), sample_rate=sr, caption=filename + '_lr')
        sr_wandb_audio = self._wandb.Audio(sr_audio.squeeze().numpy(), sample_rate=sr, caption=filename + '_pr')

        if pesq is not None and stoi is not None and sisnr is not None and lsd is not None and visqol is not None:
            self.eval_table.add_data(
                lr_wandb_audio,
                sr_wandb_audio,
                hr_wandb_audio,
                lr_wandb_spec,
                sr_audio_wandb_spec,
                hr_audio_wandb_spec,
                pesq,
                stoi,
                sisnr,
                lsd,
                visqol
            )
        else:
            self.infer_table.add_data(
                lr_wandb_audio,
                sr_wandb_audio,
                hr_wandb_audio,
                lr_wandb_spec,
                sr_audio_wandb_spec,
                hr_audio_wandb_spec,
            )

    def log_eval_table(self, commit=False):
        """
        Log the table
        """
        if self.eval_table:
            self._wandb.log({'eval_data': self.eval_table}, commit=commit)
        elif self.infer_table:
            self._wandb.log({'infer_data': self.infer_table}, commit=commit)
