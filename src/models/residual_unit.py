from tensorflow.keras import layers

class ResidualUnit(layers.Layer):
    """
    Implementation of Residual Block part of the network
    based on https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(self, channels=64, kernel_size=(3, 3), strides=(1, 1), **kwargs):
        
        """
        :params:
        1. channels -> number of filters for the residual block
        2. kernel_size -> kernel_size for second conv layer, first and third conv layer will have size (1,1)
        3. strides -> strides for second conv layer, needed because we might use different kernel_size in CONV2
        """

        super(ResidualUnit, self).__init__(**kwargs)

        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size

        ### Initialize layers needed
        self.skip_add = layers.Add()
        
        # 1x1 filters used for reducing dimensions
        self.conv1 = layers.Conv2D(self.channels//4, kernel_size=(1,1), padding='valid', strides=self.strides)
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # bottleneck layer with smaller input/output dimensions
        # kernel size dependent on input
        self.conv2 = layers.Conv2D(self.channels//4, kernel_size=self.kernel_size, padding='same')
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 1x1 filters used for restoring dimensions back
        self.conv3 = layers.Conv2D(self.channels, kernel_size=(1,1), padding='valid')
        self.batch_norm3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        # shape alignment ensuring we can add the identity input with output of CONV layer
        if self.strides != 1:
            self.conv4 = layers.Conv2D(self.channels, (1, 1), padding='valid', strides=self.strides)
        else:
            self.conv4 = None
        

    def call(self, x):
        """
        Forward pass using the residual block
        """
        
        # if stride != 1 use shortcut conv4
        if self.strides != 1:
            shortcut = self.conv4(x)
        else:
            shortcut = x

        # layer 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        # layer 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        # layer 3
        x = self.conv3(x)
        x = self.batch_norm3(x)

        # residual + identity
        x = self.skip_add([x, shortcut])
        x = self.relu3(x)

        return x
