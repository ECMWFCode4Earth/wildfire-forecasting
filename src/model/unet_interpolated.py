"""
Modification in U-Net model for low resolution gfas-frp data which requires \
interpolation due to non-whole number scaling required in the final layer.
"""
from collections import defaultdict

import torch
import torch.nn as nn

from model.unet import Model as BaseModel


class Model(BaseModel):
    """
    The primary module containing all the training functionality. It is equivalent to
    PyTorch nn.Module in all aspects.
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the
        model.

        Parameters
        ----------
        hparams : Namespace
            It contains all the major hyperparameters altering the training in some
            manner.
        """

        # init superclass
        super().__init__(hparams)

    def forward(self, x):
        """
        Forward pass
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
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return nn.functional.interpolate(self.conv(dec1), size=(1800, 3600))

    def test_epoch_end(self, outputs):
        """
        Called at the end of testing epoch to aggregate outputs.

        :param outputs: List of individual outputs of each testing step.
        :type outputs: list
        :return: Loss and logs.
        :rtype: dict
        """
        if outputs == [{}] * len(outputs):
            return {}
        avg_loss = torch.stack([x["test_loss"] for x in outputs if x]).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["test_loss"] = avg_loss

        for n in range(self.hparams.out_days):
            tensorboard_logs[f"test_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["test_loss"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["acc_test"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"mae_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["mae_test"] for x in outputs if x]]
            ).mean()

        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
        }
