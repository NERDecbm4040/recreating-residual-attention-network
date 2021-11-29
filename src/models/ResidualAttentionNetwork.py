from tensorflow import keras
from tensorflow.keras import layers

from .residual_unit import ResidualUnit
from .attention_module import AttentionModule


class ResidualAttentionNetwork(keras.Model):
    """
    Implementation of Residual Attention Network using predefined Residual and Attention Blocks
    """
    
    def __init__(self, input_shape=(32,32,3), num_class=10, channels=64, p=1, t=2, r=1, **kwargs):
        
        """
        :params:
        1. input_shape -> 3 elements tuple (height, width, channel) of input image
        2. num_class -> number of output class
        3. channels -> number of channel for 
        4. p -> number of preprocessing residual units in each stage
        5. t -> number of residual units in the trunk branch
        6. r -> number of residual units between adjacent pooling layer in the soft mask branch
        """
        
        super(ResidualAttentionNetwork, self).__init__(**kwargs)

        self.p = p
        self.t = t
        self.r = r
        
        ### Initialize layers needed

        self.conv1 = layers.Conv2D(filters=32, input_shape=input_shape, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1_activation = layers.ReLU()
        
        self.residual_unit1 = ResidualUnit(channels=128, strides=2)
        self.attention_module1 = AttentionModule(channels=128, stage=2, p=self.p, t=self.t, r=self.r)
        
        self.residual_unit2 = ResidualUnit(channels=256, strides=2)
        self.attention_module2 = AttentionModule(channels=256, stage=3, p=self.p, t=self.t, r=self.r)
        
        self.residual_unit3 = ResidualUnit(channels=512, strides=2)
        self.residual_unit4 = ResidualUnit(channels=512)
        self.residual_unit5 = ResidualUnit(channels=512)
        
        self.batch_norm2 = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.avgpool = layers.AveragePooling2D(pool_size=4, strides=1)
            
        # fully connected layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(channels//2, activation='relu')
        self.fc2 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        """
        Forward pass for the network.
        """
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv1_activation(x)

        x = self.residual_unit1(x)
        x = self.attention_module1(x)

        x = self.residual_unit2(x)
        x = self.attention_module2(x)

        x = self.residual_unit3(x)
        x = self.residual_unit4(x)
        x = self.residual_unit5(x)

        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.avgpool(x)

        # top layer
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.fc2(x)

        return x