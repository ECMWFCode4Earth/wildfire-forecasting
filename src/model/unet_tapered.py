"""
U-Net model tapered at the end for low res output.
"""
import torch
import torch.nn as nn

from model.unet import Model as BaseModel


class Model(BaseModel):
    """
    This class implements modified U-Net module by removing the up-sampling layers \
once activation resolution becomes 1/4th the resolution of input. After the removal, \
feature compression layers are added which keep the resolution constant all along. The \
layers in the tapered end additionally have skip connections similar to DenseNet. It \
is equivalent to PyTorch's nn.Module in all aspects.

    :param LightningModule: The Pytorch-Lightning module derived from nn.module with \
useful hooks
    :type LightningModule: nn.Module
    """

    def __init__(self, hparams):
        """
        Constructor for Model

        :param hparams: Holds configuration values
        :type hparams: Namespace
        """

        # init superclass
        super().__init__(hparams)
        features = self.hparams.init_features
        delattr(self, "upconv2")
        delattr(self, "upconv1")

        self.res32 = nn.Conv2d((features * 2) * 2, features * 2, kernel_size=1)
        self.res31 = nn.Conv2d((features * 2) * 2, features, kernel_size=1)
        self.res21 = nn.Conv2d(features * 2, features, kernel_size=1)

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

        dec2 = self.decoder2(dec3) + self.res32(dec3)
        dec1 = self.decoder1(dec2) + self.res21(dec2) + self.res31(dec3)
        return self.conv(dec1)
