import torch.nn as nn

# cnn_encoder = snt.nets.ConvNet2D(
#       output_channels=[128] * 4,
#       kernel_shapes=[3],
#       strides=[2, 2, 1, 1],
#       paddings=[snt.VALID],
#       activate_final=True)

class ConvNet2D(nn.Module):
    def __init__(self, activate_final=False, **kwargs):
        """
        cnn_encoder = snt.nets.ConvNet2D(
            output_channels=[128] * 4,
            kernel_shapes=[3],
            strides=[2, 2, 1, 1],
            paddings=[snt.VALID],
            activate_final=True)
        """
        super(ConvNet2D, self).__init__()
        for key in kwargs:
            if isinstance(kwargs[key], list):
        for

        self.net =
