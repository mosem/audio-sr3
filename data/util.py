import os
from copy import deepcopy

import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


def collate_fn(data):
    '''
    We should build a custom collate_fn rather than using default collate_fn,
    as the size of every sentence is different and merging sequences (including padding)
    is not supported in default.
    Args:
        data: list of dicts {'HR': hr_sig, 'SR': sr_sig, 'filename': filename}
    Return:
        dictionary of batches: {'HR': hr_sig_batch, 'SR': sr_sig_batch, 'filename': hr_filename_batch}
    '''
    lengths = [d['HR'].size(-1) for d in data]
    sig_channels = data[0]['SR'].size(0)
    max_len = max(lengths)
    hr_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['HR'].type())
    sr_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['SR'].type())

    for i in range(len(data)):
        sig_len = data[i]['HR'].size(-1)
        hr_padded[i,:,:sig_len] = data[i]['HR']
        sr_padded[i,:,:sig_len] = data[i]['SR']

    filenames = [d['filename'] for d in data]
    file_lengths = [d['length'] for d in data]


    return {'HR': hr_padded, 'SR': sr_padded, 'filename': filenames, 'length': file_lengths}


class SequentialBinSampler(torch.utils.data.Sampler):
    def __init__(self, file_lengths):
        self.file_lengths = file_lengths
        self.idx_len_pairs = [(i,length) for i,length in enumerate(self.file_lengths)]
        self.indices_sorted_by_len = [x[0] for x in sorted(self.idx_len_pairs, key=lambda x: x[1])]

    def __len__(self):
        return len(self.file_lengths)

    def __iter__(self):
        return iter(self.indices_sorted_by_len)
