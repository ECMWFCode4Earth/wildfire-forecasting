"""
Model from Forest Fire Susceptibility Modeling Using a Convolutional Neural
Network for Yunnan Province of China by Zhang et al.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
import glob

import xarray as xr
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


class GEFFDataset(Dataset):
    def __init__(self, data=None, div=3, root_dir=None, transform=None, in_depth=7):

        if not data:
            if root_dir:
                files = glob.glob(root_dir + "/**/*.*", recursive=True)
                files.sort()

                data = {}
                for file in files:
                    if "reanalysis" not in file:
                        with xr.load_dataset(file) as ds:
                            data[os.path.split(file)[-1][:-4]] = ds[
                                list(ds.data_vars)[0]
                            ]
                with xr.load_dataset("/root/net/reanalysis_fwi.nc4") as ds:
                    data[os.path.split("/root/net/reanalysis_fwi.nc4")[-1][:-4]] = ds[
                        list(ds.data_vars)[0]
                    ]

            else:
                raise ValueError("Supply at least one of data or root_dir.")

        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.div = div
        self.shape = (
            (data["reanalysis_fwi"].shape[-2] + div - 1) // div,
            (data["reanalysis_fwi"].shape[-1] + div - 1) // div,
            in_depth,
        )
        months = data["reanalysis_fwi"].time.dt.month.values - 2
        self.months = np.where(months == -1, 0, (months - 1))
        self.out_mean = np.nanmean(data["reanalysis_fwi"].values)

    def __len__(self):
        return 9 * max([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        time = (idx + (self.div ** 2)) // (self.div ** 2) - 1
        patch_n = idx - time * (self.div ** 2)
        row = patch_n // self.div
        col = patch_n - row * self.div

        lat_1 = row * self.shape[0]
        lat_2 = (row + 1) * self.shape[0]
        lon_1 = col * self.shape[1]
        lon_2 = (col + 1) * self.shape[1]

        X = np.zeros(self.shape)
        y = np.zeros(self.shape[:2])

        if 0:
            print("shape=", self.shape)
            print(f"patch_n={patch_n}, row={row}, col={col}")
            print(
                f"time {time},\nrow * self.shape[0] , (row + 1) * self.shape[0]={row * self.shape[0] , (row + 1) * self.shape[0]}",
                f"\ncol * self.shape[1] , (col + 1) * self.shape[1]={col * self.shape[1] , (col + 1) * self.shape[1]}",
            )
            print("Xshape=", X.shape, "\n", "Yshape=", y.shape)
            print("################")

        for i, c in enumerate(["cover", "fuel", "interim_tp"]):
            tmp = self.data[c].values[
                lat_1:lat_2, lon_1:lon_2,
            ]
            X[: tmp.shape[0], : tmp.shape[1], i] = tmp

        for i, c in enumerate(["interim_RH", "interim_t2m", "interim_ws10m"]):
            tmp = self.data[c][time].values[
                lat_1:lat_2, lon_1:lon_2,
            ]
            X[: tmp.shape[0], : tmp.shape[1], i + 3] = tmp

        tmp = self.data["stage"][self.months[time]].values[
            lat_1:lat_2, lon_1:lon_2,
        ]
        X[: tmp.shape[0], : tmp.shape[1], -1] = tmp

        #         ['reanalysis_danger', 'reanalysis_fwi', 'reanalysis_severity']
        tmp = self.data["reanalysis_fwi"][time].values[
            lat_1:lat_2, lon_1:lon_2,
        ]
        y[: tmp.shape[0], : tmp.shape[1]] = tmp
        y[X[:, :, 3] != X[:, :, 3]] = -1
        y = torch.from_numpy(y).reshape(-1)
        y[torch.isnan(y)] = 0

        if self.transform:
            X = self.transform(X)
        X[torch.isnan(X)] = 0
        X = X.float()

        return X, y


class Model(LightningModule):
    # TODO
    """
    The model

    Passing hyperparameters:

        >>> f=3
            x=269//f
            y=183//f
            params = dict(
                in_width=x,
                in_length=y,
                in_depth=7,
                output_size=x*y,
                drop_prob=0.5,
                learning_rate=0.001,
                conv1={"stride": 1, "kernel_size": 3, "channels": 64},
                conv2={"stride": 1, "kernel_size": 3, "channels": 128},
                conv3={"stride": 1, "kernel_size": 3, "channels": 256},
                pool1={"stride": 2, "kernel_size": 2},
                pool2={"stride": 2, "kernel_size": 2},
                fc1={"out_features": 3*x*y},
                fc2={"out_features": 2*x*y},
                fc3={"out_features": x*y},
                root_dir='/root/net/',
                epochs=20,
                optimizer_name="adam",
                batch_size=1
            )
        >>> from argparse import Namespace
        >>> hparams = Namespace(**params)
        >>> model = Model(hparams)
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the
        model.
        """

        # init superclass
        super().__init__()
        self.hparams = hparams

        conv1_padding_height = (
            (self.hparams.in_length - 1) * (self.hparams.conv1["stride"] - 1)
            + (self.hparams.conv1["kernel_size"] - 1)
        ) // 2

        conv1_padding_width = (
            (self.hparams.in_width - 1) * (self.hparams.conv1["stride"] - 1)
            + (self.hparams.conv1["kernel_size"] - 1)
        ) // 2

        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=self.hparams.in_depth,
                            out_channels=self.hparams.conv1["channels"],
                            kernel_size=self.hparams.conv1["kernel_size"],
                            stride=self.hparams.conv1["stride"],
                            padding=(conv1_padding_height, conv1_padding_width),
                        ),
                    ),
                    (
                        "pool1",
                        nn.MaxPool2d(
                            kernel_size=self.hparams.pool1["kernel_size"],
                            stride=self.hparams.pool1["stride"],
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=self.hparams.conv1["channels"],
                            out_channels=self.hparams.conv2["channels"],
                            kernel_size=self.hparams.conv2["kernel_size"],
                            stride=self.hparams.conv2["stride"],
                        ),
                    ),
                    (
                        "pool2",
                        nn.MaxPool2d(
                            kernel_size=self.hparams.pool2["kernel_size"],
                            stride=self.hparams.pool2["stride"],
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=self.hparams.conv2["channels"],
                            out_channels=self.hparams.conv3["channels"],
                            kernel_size=self.hparams.conv3["kernel_size"],
                            stride=self.hparams.conv3["stride"],
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    (
                        "conv4",
                        nn.Conv2d(
                            in_channels=self.hparams.conv3["channels"],
                            out_channels=self.hparams.conv4["channels"],
                            kernel_size=self.hparams.conv4["kernel_size"],
                            stride=self.hparams.conv4["stride"],
                        ),
                    ),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        in_features = self.get_conv_fts()
        print(in_features)

        self.fc_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            in_features=in_features,
                            out_features=self.hparams.fc1["out_features"],
                        ),
                    ),
                    #                    ("relu1", nn.ReLU()),
                    #                    ("drop1", nn.Dropout(self.hparams.drop_prob)),
                    #                    (
                    #                        "fc2",
                    #                        nn.Linear(
                    #                            in_features=self.hparams.fc1["out_features"],
                    #                            out_features=self.hparams.fc2["out_features"],
                    #                        ),
                    #                    ),
                    #                    ("relu2", nn.ReLU()),
                    #                    ("drop2", nn.Dropout(self.hparams.drop_prob)),
                    #                    (
                    #                        "fc3",
                    #                        nn.Linear(
                    #                            in_features=self.hparams.fc2["out_features"],
                    #                            out_features=self.hparams.fc3["out_features"],
                    #                        ),
                    #                    ),
                    ("relu2", nn.ReLU()),
                    #                    ("drop2", nn.Dropout(self.hparams.drop_prob)),
                    (
                        "fc4",
                        nn.Linear(
                            in_features=self.hparams.fc3["out_features"],
                            out_features=self.hparams.output_size,
                        ),
                    ),
                ]
            )
        )

    def get_conv_fts(self):
        return self.conv_block(
            torch.zeros(
                1, self.hparams.in_depth, self.hparams.in_width, self.hparams.in_length
            )
        ).shape[1]

    def forward(self, x):
        """
        Forward pass
        """
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Called inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x) + self.data.out_mean
        pre_loss = (y_hat - y) ** 2
        mask = y > -0.5
        loss = pre_loss[mask].mean()
        tensorboard_logs = {"train_loss": loss}
        # wandb.log(tensorboard_logs)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Called inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x) + self.data.out_mean
        pre_val_loss = (y_hat - y) ** 2
        mask = y > -0.5
        val_loss = pre_val_loss[mask].mean()
        n_correct_pred = torch.sum(((y - y_hat).abs() < 5)[mask]).item()
        return {
            "val_loss": val_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": mask.sum().item(),
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) + self.data.out_mean
        pre_test_loss = (y_hat - y) ** 2
        mask = y > -0.5
        test_loss = pre_test_loss[mask].mean()
        n_correct_pred = torch.sum(((y - y_hat).abs() < 5)[mask]).item()
        return {
            "test_loss": test_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": mask.sum().item(),
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc}
        # wandb.log(tensorboard_logs)
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "out_mean": self.data.out_mean,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        tensorboard_logs = {
            "test_loss": avg_loss,
            "test_acc": test_acc,
            "out_mean": self.data.out_mean,
        }
        # wandb.log(tensorboard_logs)
        return {"test_loss": avg_loss, "log": tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return optimizers and learning rate schedulers.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
        )
        self.data = GEFFDataset(
            root_dir=self.hparams.root_dir,
            transform=transform,
            in_depth=self.hparams.in_depth,
        )
        self.train_data, self.test_data = torch.utils.data.random_split(
            self.data,
            [len(self.data) * 8 // 10, len(self.data) - len(self.data) * 8 // 10],
        )

    #            [180,len(self.data)-180]

    def train_dataloader(self):
        log.info("Training data loader called.")
        return DataLoader(
            self.train_data, batch_size=self.hparams.batch_size, num_workers=4
        )

    def val_dataloader(self):
        log.info("Validation data loader called.")
        return DataLoader(
            self.test_data, batch_size=self.hparams.batch_size, num_workers=4
        )

    def test_dataloader(self):
        log.info("Test data loader called.")
        return DataLoader(
            self.test_data, batch_size=self.hparams.batch_size, num_workers=4
        )

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters defined here will be available to model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument("--in_width", default=183, type=int)
        parser.add_argument("--in_length", default=269, type=int)
        parser.add_argument("--in_depth", default=269, type=int)
        parser.add_argument("--output_size", default=49227, type=int)
        parser.add_argument("--drop_prob", default=0.5, type=float)
        parser.add_argument("--learning_rate", default=0.001, type=float)

        parser.add_argument(
            "--conv1",
            default='{"stride": 1, "kernel_size": 3, "channels": 64}',
            type=json.loads,
        )
        parser.add_argument(
            "--conv2",
            default='{"stride": 1, "kernel_size": 3, "channels": 128}',
            type=json.loads,
        )
        parser.add_argument(
            "--conv3",
            default='{"stride": 1, "kernel_size": 3, "channels": 256}',
            type=json.loads,
        )

        parser.add_argument(
            "--pool1", default='{"stride": 2, "kernel_size": 2}', type=json.loads
        )
        parser.add_argument(
            "--pool2", default='{"stride": 2, "kernel_size": 2}', type=json.loads
        )

        parser.add_argument("--fc1", default='{"out_features": 49227}', type=json.loads)
        parser.add_argument("--fc2", default='{"out_features": 49227}', type=json.loads)
        parser.add_argument("--fc3", default='{"out_features": 49227}', type=json.loads)

        # data
        parser.add_argument("--root_dir", default="/root/net/", type=str)

        # training params (opt)
        parser.add_argument("--epochs", default=20, type=int)
        # Test split ratio (opt)
        parser.add_argument("--split", default=0.2, type=int)
        # parser.add_argument("--optimizer_name", default="adam", type=str)
        parser.add_argument("--batch_size", default=1, type=int)
        return parser
