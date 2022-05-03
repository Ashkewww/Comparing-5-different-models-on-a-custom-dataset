import os
import numpy as np

import tensorflow as tf
from tensorflow import keras


train_folder = os.path.join('data','train images')
test_folder = os.path.join('data', 'test images')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_folder,
    validation_split=0.1,
    subset='training',
    seed=42,
    image_size=(250,250),
    label_mode='categorical',
    batch_size=16,
    shuffle=True
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    train_folder,
    validation_split=0.1,
    subset='validation',
    seed=42,
    image_size=(250,250),
    batch_size=16,
    label_mode='categorical',
    shuffle=True
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_folder,
    image_size=(250,250),
    label_mode='categorical',
    shuffle=False)

class_names = test_ds.class_names




