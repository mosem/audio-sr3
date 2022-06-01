'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase,sampler=None,collate_fn=None):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        batch_size = dataset_opt['batch_size'] if 'batch_size' in dataset_opt else 1
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,sampler=sampler,collate_fn=collate_fn)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    # from data.LRHR_dataset import LRHRDataset as D
    from data.LRHR_audio_dataset import LRHRAudioDataset as D
    dataset = D(json_dir=dataset_opt['json_dir'],
                lr_sr=dataset_opt['lr_sr'],
                hr_sr=dataset_opt['hr_sr'],
                segment=dataset_opt['segment'] if 'segment' in dataset_opt else None,
                stride=dataset_opt['stride'] if 'stride' in dataset_opt else None,
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
