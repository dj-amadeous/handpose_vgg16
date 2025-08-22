import tensorflow as tf
from tensorflow.keras.applications import VGG16
import glob
import numpy as np
from data_generator import DataGenerator
from scipy.io import loadmat

INPUT_SHAPE = (180, 180, 3)
IMAGE_ROOT = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"
IMAGE_TRAIN = IMAGE_ROOT + "/train"
LABEL_TRAIN = IMAGE_ROOT + "/train/joint_data.mat"
LABEL_TEST = IMAGE_ROOT + "/test/joint_data.mat"


# Load the VGG16 model with pre-trained ImageNet weights
model = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)






# create image dataset for retraining
nyu_dataset_train = DataGenerator(IMAGE_TRAIN, )

# 1. add a linear layer to the end of VGG16
# 2. add a classifier layer with the same number of nodes as joints
# 3. train end layers

# Perform inference
predictions = model.predict(img_array)

