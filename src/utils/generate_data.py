import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as preprocessing
from tensorflow.keras import layers
from tensorflow.data import Dataset
import os
import json
import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalize(x):
    x = x.astype("float32")
    x = x / 255.0
    return x

def preprocess_dataset(x_train, y_train, x_test, y_test):
    '''
    1. Normalize images
    2. Set label as one-hot encoded vector
    3. Subtract pixel mean
    
    :input_param:
    x_train, y_train, x_test, y_test
    
    :return:
    preprocessed x_train, y_train, x_test, y_test
    '''

    # normalize pixel value
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # label one hot encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # mean centering
    x_train = x_train - x_train.mean()
    x_test = x_test - x_test.mean()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test


def preprocess_dataset_tensor(x_train, y_train, x_test, y_test):
    '''
    1. Normalize images
    2. Set label as one-hot encoded vector
    3. Subtract pixel mean
    
    :input_param:
    x_train, y_train, x_test, y_test
    
    :return:
    preprocessed x_train, y_train, x_test, y_test
    '''

    # normalize pixel value
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # label one hot encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # mean centering
    x_train = x_train - x_train.mean()
    x_test = x_test - x_test.mean()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    #create validation data from training data
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.20,stratify=y_train,random_state=42,shuffle=True)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_cifar10(**kwargs):
    '''
    Load CIFAR data with preprocessing and image datagen
    
    :param:
    Input for ImageDataGenerator, example:
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2

    :return:
    x_train, y_train, x_test, y_test, datagen
    
    '''

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train, y_train, x_test, y_test = preprocess_dataset(x_train, y_train, x_test, y_test)

    datagen = preprocessing.image.ImageDataGenerator(**kwargs)
    datagen.fit(x_train)

    return x_train, y_train, x_test, y_test, datagen    #test - 20% of total dataset
    test_dataset = Dataset.from_tensor_slices((x_test, y_test))
    #shuffle, batch, and prefetch data
    test_dataset = test_dataset.shuffle(buffer_size=shuffle_buffer).batch(batch_size).prefetch(buffer_size=autotune)


    return train_dataset, validation_dataset, test_dataset