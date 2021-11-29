import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as preprocessing

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

    return x_train, y_train, x_test, y_test, datagen