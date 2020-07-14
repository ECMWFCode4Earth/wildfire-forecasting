"""
The original U-Net model with a downsampling layer at the end to match with the
FWI-Reanalysis resolution.
"""
import torch.nn as nn

from model.unet import Model as BaseModel


class Model(BaseModel):
    """This class implements modified U-Net module by downsampling the output to match \
with the resolution of fwi-reanalysis. It is equivalent to PyTorch's nn.Module in all \
aspects.

    :param LightningModule: The Pytorch-Lightning module derived from nn.module with
useful hooks
    :type LightningModule: nn.Module
    """

    def __init__(self, hparams):
        """Constructor for Model.

        :param hparams: Holds configuration values
        :type hparams: Namespace
        """

        # init superclass
        super().__init__(hparams)
        out_channels = self.hparams.out_days
        features = self.hparams.init_features

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=4, stride=4,
        )
