from tensorflow.keras import layers

from .residual_unit import ResidualUnit

class TrunkBranch(layers.Layer):
    """
    Generating trunk branch

    Trunk branch performs feature processing
    can be adapted by any state-of-the-art network structures.
    """

    def __init__(self, channels, t=2, **kwargs):
        """
        :params:
        1. channels -> number of channel for each residual units
        2. t -> number of trunk branches
        """

        super(TrunkBranch, self).__init__(**kwargs)
        
        self.channels = channels
        self.t = t
        self.residual_units = []
        
        for i in range(self.t):
            setattr(self, f'residual_units{i}', ResidualUnit(self.channels))
            
    def call(self, x):
        """
        Forward pass the trunk branch
        """
        
        # Feature generation
        for i in range(self.t):
            x = getattr(self, f'residual_units{i}')(x)
        
        return x


class MaskBranch(layers.Layer):
    """
    Generating soft mask branch
    
    Mask branch uses bottom-up top-down structure 
    learn same size  mask M(x) that soft weight output features T(x)
    """

    def __init__(self, channels, r=1, stage=0, **kwargs):
        """
        :params:
        1. channels -> number of channel for each residual units
        2. r -> number of residual units between adjacent pooling layer in the soft mask branch
        3. stage -> current attention module stage
        """
        
        super(MaskBranch, self).__init__(**kwargs)

        self.r = r
        self.channels = channels
        self.num_of_pool = 3-stage

        # downsampling
        for i in range(self.num_of_pool):
            setattr(self, f'maxpool{i}', layers.MaxPool2D((2, 2)))
            for j in range(self.r):
                setattr(self, f'residual_units{i}_{j}', ResidualUnit(self.channels))
            setattr(self, f'skip_residual_units{i}', ResidualUnit(self.channels))

        self.maxpooling = layers.MaxPool2D((2, 2))
        
        # middle upsampling -> interpolation
        for k in range(2 * self.r):
            setattr(self, f'nested_residual_units{k}', ResidualUnit(self.channels))
        self.interpolation = layers.UpSampling2D(size=(2, 2))

        # last upsampling
        for l in range(self.num_of_pool):
            setattr(self, f'add{i}', layers.Add())
            for m in range(self.r):
                setattr(self, f'residual_units{l+i+1}_{m}', ResidualUnit(self.channels))
            setattr(self, f'interpolation{l}', layers.UpSampling2D(size=(2, 2)))
            
        ## Output
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv1 = layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
        
    def call(self, x):
        """
        Forward pass the mask branch
        """

        # feed-forward sweep and top-down feedback

        ## encoder
        ### first down sampling
        for i in range(self.num_of_pool):
            x = getattr(self, f'maxpool{i}')(x)
            for j in range(self.r):
                x = getattr(self, f'residual_units{i}_{j}')(x)
            setattr(self, f'skip_residual{i}', getattr(self, f'skip_residual_units{i}')(x))

        ## decoder
        ### middle upsampling
        x = self.maxpooling(x)
        for k in range(2 * self.r):
            x = getattr(self, f'nested_residual_units{k}')(x)
        x = self.interpolation(x)
        
        ### last upsampling
        for l in range(self.num_of_pool):
            x = getattr(self, f'add{i}')([x, getattr(self, f'skip_residual{self.num_of_pool-l-1}')])
            for m in range(self.r):
                x = getattr(self, f'residual_units{l+i+1}_{m}')(x)
            x = getattr(self, f'interpolation{l}')(x)
            
        ## Output
        x = self.batch_norm1(x)
        x = self.relu1(x)
        
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x



class LocalConvMaskBranch(layers.Layer):
    """
    Generating soft mask branch
    
    Mask branch uses bottom-up top-down structure 
    learn same size  mask M(x) that soft weight output features T(x)
    """

    def __init__(self, channels, r=1, stage=0, **kwargs):
        """
        :params:
        1. channels -> number of channel for each residual units
        2. r -> number of residual units between adjacent pooling layer in the soft mask branch
        3. stage -> current attention module stage
        """
        
        super(LocalConvMaskBranch, self).__init__(**kwargs)

        self.r = r
        self.channels = channels
        self.num_of_pool = 3-stage

        # # downsampling
        # for i in range(self.num_of_pool):
        #     setattr(self, f'maxpool{i}', layers.MaxPool2D((2, 2)))
        #     for j in range(self.r):
        #         setattr(self, f'residual_units{i}_{j}', ResidualUnit(self.channels))
        #     setattr(self, f'skip_residual_units{i}', ResidualUnit(self.channels))

        # self.maxpooling = layers.MaxPool2D((2, 2))
        
        # # middle upsampling -> interpolation
        # for k in range(2 * self.r):
        #     setattr(self, f'nested_residual_units{k}', ResidualUnit(self.channels))
        # self.interpolation = layers.UpSampling2D(size=(2, 2))

        # # last upsampling
        # for l in range(self.num_of_pool):
        #     setattr(self, f'add{i}', layers.Add())
        #     for m in range(self.r):
        #         setattr(self, f'residual_units{l+i+1}_{m}', ResidualUnit(self.channels))
        #     setattr(self, f'interpolation{l}', layers.UpSampling2D(size=(2, 2)))
            
        ## Output
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv1 = layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
        
    def call(self, x):
        """
        Forward pass the mask branch
        """

        # feed-forward sweep and top-down feedback

        ## encoder
     
        ## Output
        x = self.batch_norm1(x)
        x = self.relu1(x)
        
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x