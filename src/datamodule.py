import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import re
import cv2
from random import shuffle

from utils import compute_distances
from utils import *
from layers import *

import pytorch_lightning as pl


class Marker2SkelDataModule(pl.LightningDataModule):
    
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    def prepare_data(self):
        dataset_path = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        train_data = np.load(dataset_path+'/train.npz',allow_pickle=True)
        train_files = train_data['c3d']
        valid_data = np.load(dataset_path+'/validation.npz',allow_pickle=True)
        valid_files = valid_data['c3d']
        test_data = np.load(dataset_path+'/test.npz',allow_pickle=True)
        test_files = test_data['c3d']
        print('Files have been loaded')

        num_motions = len(train_files)
        print('Loaded Training samples: ' + str(num_motions))

        self.train_files = train_files
        self.validation_files = valid_files
        self.test_files = test_files

    def train_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'train.npz')

        transform = transforms.ToTensor()

        dataset = poseDATA(self.FLAGS, dataset_path, self.train_files, transform)

        marker2skel_train = FastDataLoader(dataset, shuffle=True, pin_memory=True,
                                           num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return marker2skel_train

    def val_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'validation.npz')

        transform = transforms.ToTensor()

        dataset = poseDATA(self.FLAGS, dataset_path, self.validation_files,transform)

        marker2skel_val = FastDataLoader(dataset, shuffle=False, pin_memory=True,
                                         num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return marker2skel_val

    def test_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'test.npz')

        transform = transforms.ToTensor()

        dataset = poseDATA(self.FLAGS, dataset_path, self.test_files, transform)

        marker2skel_test = FastDataLoader(dataset, shuffle=False, pin_memory=True,
                                          num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return marker2skel_test

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class poseDATA(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, FLAGS, data_path, input_list, transform):

        self.data_path = data_path
        self.data, self.labels = [], []
        self.num_motions = len(input_list)
        self.list_images = input_list
        self.list_masks = []
        self.dataset_iterations = FLAGS.dataset_iterations
        self.split_size = self.num_motions // FLAGS.num_workers
        self.starting_point = -1
        self.idx = -1
        self.FLAGS = FLAGS
        self.transform = transform

        self.files = np.load(self.data_path,allow_pickle=True)

    def __len__(self):
        return self.num_motions * self.dataset_iterations * self.FLAGS.batch_size // self.FLAGS.n_epochs

    def __getitem__(self, idx):

        # if(self.starting_point == -1):
        #     self.starting_point = (
        #         torch.utils.data.get_worker_info().id) * self.split_size

        # if(self.idx == -1):
        #     next_id = self.starting_point
        # else:
        #     next_id = (self.idx + 1)

        # if(self.idx >= self.starting_point + self.split_size - 1):
        #     self.starting_point = (
        #         torch.utils.data.get_worker_info().id) * self.split_size
        #     self.idx = self.idx - self.split_size
        # else:
        #     self.idx = next_id

        marker_data = self.files['c3d']
        skeletons = self.files['skeleton']
        smpl_skeletons = self.files['smpl']

        print("Shape marker:", marker_data[idx].shape)

        if(self.transform):
            markers = self.transform(marker_data)

        label = {'markers': markers,
                #  'rm': rm, 
                 'skeletons': skeletons,
                 'smpl': smpl_skeletons}

        return markers, skeletons, smpl_skeletons, label

