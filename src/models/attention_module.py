from tensorflow.keras import layers

from .residual_unit import ResidualUnit
from .attention_branch import TrunkBranch, MaskBranch, LocalConvMaskBranch

class AttentionModule(layers.Layer):
    """
    Implementation of Attention Block part of the network
    based on https://arxiv.org/abs/1704.06904
    """

    def __init__(self, channels=64, stage=0, p=1, t=2, r=1, learning_type='arl',mask_type='enc-dec', **kwargs):
        
        """
        :params:
        1. channels -> number of channel for each residual units
        2. stage -> current attention module stage
        3. p -> number of preprocessing residual units in each stage
        4. t -> number of residual units in the trunk branch
        5. r -> number of residual units between adjacent pooling layer in the soft mask branch
        6. learning_type -> arl for Attention Residual Learning, nal for Naive Attention Learning
        """
        
        super(AttentionModule, self).__init__(**kwargs)

        # Initialize hyperparameters
        self.p = p
        self.t = t
        self.r = r
        self.channels = channels
        self.stage = stage
        self.learning_type = learning_type
        self.mask_type = mask_type

        # First Residual Block
        for i in range(2*self.p):
            setattr(self, f'residual_units{i}', ResidualUnit(self.channels))
            

        # Generate Mask and Trunk branches
        if mask_type == 'enc-dec':
            self.mask_branch = MaskBranch(self.channels, r=self.r, stage=self.stage)
        else:
            self.mask_branch = LocalConvMaskBranch(self.channels, r=self.r, stage=self.stage)
        self.trunk_branch = TrunkBranch(self.channels, t=self.t)

        # used for mask branch layers
        self.multiply = layers.Multiply()
        self.add = layers.Add()

    def call(self, x):
        """
        Forward pass using soft mask branch and trunk branch.
        
        Output Hi,c(x) = (1 + Mi,c(x)) ∗ Fi,c(x) for ARL
        Output Hi,c(x) = Mi,c(x) ∗ Fi,c(x) for NAL
        """

        for i in range(self.p):
            x = getattr(self, f'residual_units{i}')(x)

        x_mask = self.mask_branch(x)
        x_trunk = self.trunk_branch(x)

        # Calculate Attention: 
        if self.learning_type == 'arl':
            # Residual Attention = (1 + output_soft_mask) * output_trunk
            x = self.multiply([x_mask, x_trunk])
            x = self.add([x, x_trunk])
        elif self.learning_type == 'nal':
            # Naive Attention = (output_soft_mask * output_trunk)
            x = self.multiply([x_mask, x_trunk])
        
        for j in range(self.p):
            x = getattr(self, f'residual_units{i+j+1}')(x)

        return x