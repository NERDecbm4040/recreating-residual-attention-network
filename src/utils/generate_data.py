import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

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
    preprocessed x_train, y_train, x_val, y_val, x_test, y_test
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train, random_state=42, shuffle=True)

    return x_train, y_train, x_val, y_val, x_test, y_test

def get_cifar10(*preproc_layers):
    '''
    Load CIFAR 10 data with preprocessing and image datagen
    
    :param:
    List of data augmentation layers, example:
        [
            layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ]

    :return:
    x_train, y_train, x_test, y_test, datagen
    
    '''

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_dataset(x_train, y_train, x_test, y_test)

    input_shape = x_train.shape[1:]
    num_class = y_train.shape[1]

    # data augmentation layers
    data_aug = keras.Sequential(
        *preproc_layers
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return input_shape, num_class, train_ds, val_ds, test_ds, data_aug