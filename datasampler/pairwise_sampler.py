from __future__ import absolute_import, division

import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps


class RandomStretch(object):
    '''
    deal with image data to pairs
    '''
    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        '''
        make the instanse be callable
        return img's resize
        '''
        # scale the image randomly
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float)*scale).astype(int)

        # scale the image using bilinear or bicubic method
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        if self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)


class PairwiseSampler(Dataset):
    '''
    sampling for training dataset 
    '''
    def __init__(self, seq_dataset, **kargs):
        super(PairwiseSampler, self).__init__()
        self.cfg = self.parse_args(**kargs)

        self.seq_dataset = seq_dataset
        self.indices =np.random.permunation(len(seq_dataset))
        # augmentation for examplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_z - 8),
            RandomCrop(self.cfg.instance_sz - 2*8),
            CenterCrop(self.cfg.examplar_sz),
            ToTensor()
        ])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2*8),
            ToTensor()
        ])

    def parse_args(self, **kargs):
        # default paprameters
        cfg = {
            'pairs_per_seq': 10,
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5
        }

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def __getitem__(self, index):
        '''
        replement for __getitem__()
        return: examplar_image and  instance_image
        '''
        index = self.indices[index % len(self.seq_dataset)]
        img_files, anno = self.seq_dataset[index]

        # select pair frames from the seq_dataset
        rand_z, rand_x = self._sample_pair(len(img_files))
        examplar_image = Image.open(img_files[rand_z])
        instance_image = Image.open(img_files[rand_x])
        examplar_image = self._crop_and_resize(examplar_image, anno[rand_z])
        instance_image = self._crop_and_resize(instance_image, anno[rand_x])
        examplar_image = 255.0 * self.transform_z(examplar_image)
        instance_image = 255 * self.transform_x(instance_image)

        return examplar_image, instance_image

    
    def __len__(self):
        return  self.cfg.pair_per_seq * len(self.seq_dataset)

    def _sample_pair(self, n):
        '''
        the indices of examplar and instance imagea from a sequence frame
        args:
            n: teh number of a sequence frames
            return: the examplar index and the instance index
        '''
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.cfg.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist
        return rand_z, rand_x

    def _crop_and_resize(self, image, box):
        '''
        crop image and resize it.
        args:
            image: to be converted
            box: the annotation of the image
            return: the patch of the image
        '''
        # convert box to the 0-indexed and center based
        box = np.array([box[0]-1 + (box[2]-1)/2, box[1]-1 + (box[3] - 1)/2], box[2], box[3], dtype = np.float32)
        center, target_sz = box[:2], box[2:]

        # examplar and search size
        context = self.cfg.context*np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.cfg.instance_sz / self.cfg.examplar_sz

        # convert the box to corners(0-indexes)
        size = round(x_sz)
        corners = np.concatenate(np.round(center - (size-1)/2), np.round(center - (size - 1)/2) + size)
        corners = np.round(corners).astype(int)

        # add padding if necessary
        pads = np.concatenate((-corners[:2], corners[2:]-image.size))
        npad = max(0, int(pads.max()))

        if npad > 0:
            avg_color = ImageStat.Stat(image).mean

            # PIL doesn't support float RGB image
            avg_color = tuple(int(int(round(c))) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        # resize to instance_sz
        out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)

        return patch


if __name__ == "__main__":
    # test RandomStretch
    rstretch = RandomStretch(max_stretch=0.05)
    img = Image.open('../dataset/cat.jpg')
    print('row image size: ', img.size)
    print('pairwise test, RandomStetch: ', rstretch(img))
    print('permunation: ', np.random.permutation(10))