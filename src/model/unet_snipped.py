"""
Modification in U-Net model for fwi-reanalysis. The upsampling layers towards the end
are removed before the activation resolution gets higher than the output resolution.
"""
import torch
import torch.nn as nn

from model.unet import Model as BaseModel


class Model(BaseModel):
    """
    This class implements modified U-Net module by removing the up-sampling layers \
once activation resolution matches with the resolution of fwi-reanalysis. It is \
equivalent to PyTorch's nn.Module in all aspects.

    :param LightningModule: The Pytorch-Lightning module derived from nn.module with \
useful hooks
    :type LightningModule: nn.Module
    """

    def __init__(self, hparams):
        """
        Constructor for Model.

        :param hparams: Holds configuration values
        :type hparams: Namespace
        """

        # init superclass
        super().__init__(hparams)
        out_channels = 1
        features = self.hparams.init_features

        self.conv = nn.Conv2d(
            in_channels=4 * features, out_channels=out_channels, kernel_size=1,
        )

    def forward(self, x):
        """
        Does the forward pass on the model.

        :param x: Input tensor batch.
        :type x: torch.Tensor
        :return: Output activations.
        :rtype: torch.Tensor
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        return self.conv(dec3)
