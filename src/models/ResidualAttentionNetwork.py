from scipy.sparse import data
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .residual_unit import ResidualUnit
from .attention_module import AttentionModule


class ResidualAttentionNetwork(keras.Model):
    """
    Implementation of Residual Attention Network using predefined Residual and Attention Blocks
    """
    
    def __init__(
        self, 
        input_shape=(32,32,3), 
        num_class=10, 
        data_augmentation=[
            layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ],
        channels=[32, 64, 128, 256, 512], 
        num_blocks=[1,1,1],
        dropout=None,
        regularization=0.01,
        p=1, t=2, r=1, 
        learning_type='arl', 
        **kwargs
    ):
        
        """
        :params:
        1. input_shape -> 3 elements tuple (height, width, channel) of input image
        2. num_class -> number of output class
        3. data_augmentation -> list containing data augmentation as a layer
        4. channels -> list of number of channel for each layer: 
            convolutional layer, 
            residual-attention stage1, 
            residual-attention stage2, 
            residual-attention stage3, 
            pre-activation residual units
        5. num_blocks -> number of attention module in each residual-attention stage
        6. dropout -> Float between 0 and 1, Fraction of the input units to drop
        7. regularization -> L2 regularizer value

        # Attention Module Parameters
        8. p -> number of preprocessing residual units in each stage
        9. t -> number of residual units in the trunk branch
        10. r -> number of residual units between adjacent pooling layer in the soft mask branch
        11. learning_type -> arl for Attention Residual Learning, nal for Naive Attention Learning
        """
        
        super(ResidualAttentionNetwork, self).__init__(**kwargs)

        self.p = p
        self.t = t
        self.r = r
        self.learning_type = learning_type
        self.channels = channels
        self.num_blocks = num_blocks
        
        ### Initialize layers needed

        # data augmentation layers
        self.data_augmentation = keras.Sequential(data_augmentation)

        # Convolutional Layer
        self.conv1 = layers.Conv2D(filters=self.channels[0], input_shape=input_shape, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1_activation = layers.ReLU()
        self.maxpool1 = layers.MaxPool2D(pool_size=2, strides=1, padding='same')
        
        # Residual-Attention stage 1
        self.residual_unit1 = ResidualUnit(channels=self.channels[1], strides=2)
        self.attention_module1 = []
        for _ in range(self.num_blocks[0]):
            self.attention_module1.append(AttentionModule(channels=self.channels[1], stage=1, p=self.p, t=self.t, r=self.r, learning_type=self.learning_type))
        
        # Residual-Attention stage 2
        self.residual_unit2 = ResidualUnit(channels=self.channels[2], strides=2)
        self.attention_module2 = []
        for _ in range(self.num_blocks[1]):
            self.attention_module2.append(AttentionModule(channels=self.channels[2], stage=2, p=self.p, t=self.t, r=self.r, learning_type=self.learning_type))

        # Residual-Attention stage 3
        self.residual_unit3 = ResidualUnit(channels=self.channels[3], strides=2)
        self.attention_module3 = []
        for _ in range(self.num_blocks[2]):
            self.attention_module3.append(AttentionModule(channels=self.channels[3], stage=3, p=self.p, t=self.t, r=self.r, learning_type=self.learning_type))
        
        # Pre-activation Residual Units
        self.residual_unit4 = ResidualUnit(channels=self.channels[4], strides=2)
        self.residual_unit5 = ResidualUnit(channels=self.channels[4])
        self.residual_unit6 = ResidualUnit(channels=self.channels[4])
        
        self.batch_norm2 = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.avgpool = layers.AveragePooling2D(pool_size=2, strides=1)
        
        # fully connected layers
        self.flatten = layers.Flatten()
        if dropout:
            self.dropout = layers.Dropout(dropout)
        else:
            self.dropout = None
        # Use L2 regularization
        self.fc = layers.Dense(num_class, kernel_regularizer=regularizers.l2(regularization), activation='softmax')

    def call(self, x):
        """
        Forward pass for the network.
        """
        
        x = self.data_augmentation(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv1_activation(x)
        x = self.maxpool1(x)

        x = self.residual_unit1(x)
        for am_s1 in self.attention_module1:
            x = am_s1(x)

        x = self.residual_unit2(x)
        for am_s2 in self.attention_module2:
            x = am_s2(x)

        x = self.residual_unit3(x)
        for am_s3 in self.attention_module3:
            x = am_s3(x)

        x = self.residual_unit4(x)
        x = self.residual_unit5(x)
        x = self.residual_unit6(x)

        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.avgpool(x)

        # top layer
        x = self.flatten(x) 
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)

        return x

class Attention56(ResidualAttentionNetwork):
    """
    Implementation of Attention 56 using Residual Attention Network Model
    """
    def __init__(
        self, input_shape=(32,32,3), num_class=10, dropout=0.4, regularization=0.01, learning_type='arl',
        data_augmentation=[
            layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ]
    ):
        
        """
        :params:
        1. input_shape -> 3 elements tuple (height, width, channel) of input image
        2. num_class -> number of output class
        3. dropout -> Float between 0 and 1, Fraction of the input units to drop
        4. regularization -> L2 regularizer value
        5. learning_type -> arl for Attention Residual Learning, nal for Naive Attention Learning
        6. data_augmentation -> list containing data augmentation as a layer
        """

        super(Attention56, self).__init__(
            input_shape=input_shape,
            num_class=num_class,
            data_augmentation=data_augmentation,
            # Fix number of channels and attention blocks
            channels=[64, 256, 512, 1024, 2048],
            num_blocks=[1, 1, 1],
            dropout=dropout,
            regularization=regularization,
            learning_type=learning_type
        )

class Attention92(ResidualAttentionNetwork):
    """
    Implementation of Attention 92 using Residual Attention Network Model
    """
    def __init__(
        self, input_shape=(32,32,3), num_class=10, dropout=0.4, regularization=0.01, learning_type='arl',
        data_augmentation=[
            layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ]
    ):
        
        """
        :params:
        1. input_shape -> 3 elements tuple (height, width, channel) of input image
        2. num_class -> number of output class
        3. dropout -> Float between 0 and 1, Fraction of the input units to drop
        4. regularization -> L2 regularizer value
        5. learning_type -> arl for Attention Residual Learning, nal for Naive Attention Learning
        6. data_augmentation -> list containing data augmentation as a layer
        """

        super(Attention92, self).__init__(
            input_shape=input_shape,
            num_class=num_class,
            data_augmentation=data_augmentation,
            # Fix number of channels and attention blocks
            channels=[64, 256, 512, 1024, 2048],
            num_blocks=[1, 2, 3],
            dropout=dropout,
            regularization=regularization,
            learning_type=learning_type
        )