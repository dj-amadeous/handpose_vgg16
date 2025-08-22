import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(180, 180), shuffle=True):
        '''
        constructor for creating a data generator. as of 04/15/2025 constructor couples generator heavily with NYU
        that must be edited going forward. TODO decouple from nyu dataset 04/15/2025
            can be done by making image path more generic, other datasets don't have consisting naming in their files.
        :param image_paths:
        :param labels:
        :param batch_size:
        :param image_size:
        :param shuffle:
        '''
        self.image_paths = glob.glob(image_paths + "/depth_1_*.png") # this constructor couples the data generator heavily to NYU structure and should be changed
        self.order_image_paths()
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Overloaded function: number of batches in epoch
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        # Overloaded function: runs every time epoch ends
        self.indices = np.arange(self.image_paths)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, item):
        # Overloaded function: generates one batch of data
        batch_indices = self.indices[item * self.batch_size:(item + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]

        images = [self._load_image(path) for path in batch_image_paths]
        return np.stack(images), batch_labels

    def _load_image(self, path):
        # Custom function: loads an image with cv2 and interpolates slash normalizes to 0-1
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        return image

    def order_image_paths(self):
       self.image_paths.sort(key=lambda st: int(st.split('_')[-1].split('.')[0]))


