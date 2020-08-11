"""
The dataset class to be used with fwi-forcings and gfas-frp data.
"""
from glob import glob
from collections import defaultdict

import xarray as xr
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox

import torch

from dataloader.base_loader import ModelDataset as BaseDataset


class ModelDataset(BaseDataset):
    """
    The dataset class responsible for loading the data and providing the samples for
    training.
    """

    def __init__(
        self,
        out_var=None,
        out_mean=None,
        forecast_dir=None,
        forcings_dir=None,
        reanalysis_dir=None,
        frp_dir=None,
        hparams=None,
        **kwargs,
    ):
        """
        Constructor for the ModelDataset class

        :param out_var: Variance of the output variable, defaults to None
        :type out_var: float, optional
        :param out_mean: Mean of the output variable, defaults to None
        :type out_mean: float, optional
        :param forecast_dir: The directory containing the FWI-Forecast data, defaults \
to None
        :type forecast_dir: str, optional
        :param forcings_dir: The directory containing the FWI-Forcings data, defaults \
to None
        :type forcings_dir: str, optional
        :param reanalysis_dir: The directory containing the FWI-Reanalysis data, \
to defaults to None
        :type reanalysis_dir: str, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        super().__init__(
            out_var=out_var,
            out_mean=out_mean,
            forecast_dir=forecast_dir,
            forcings_dir=forcings_dir,
            reanalysis_dir=reanalysis_dir,
            frp_dir=None,
            hparams=hparams,
            **kwargs,
        )

        # Number of input and prediction days
        assert (
            self.hparams.in_days > 0 and self.hparams.out_days > 0
        ), "The number of input and output days must be > 0."
        self.hparams.in_days = self.hparams.in_days
        self.hparams.out_days = self.hparams.out_days
        self.hparams.thresh = self.hparams.out_mad / 2

        # Generate the list of all valid files in the specified directories
        inp_time = (
            lambda x: int(x.split("_20")[1][:2]) * 10000
            + int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) * 100
            + int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:])
        )
        inp_files = sorted(
            sorted(glob(f"{forcings_dir}/ECMWF_FO_20*.nc")),
            # Extracting the month and date from filenames to sort by time.
            key=inp_time,
        )
        out_time = lambda x: int(x[-8:-6]) * 100 + int(x[-5:-3])
        out_files = sorted(
            sorted(glob(f"{frp_dir}/FRP_20??_??.nc")),
            # Extracting the year and month from filenames to sort by time.
            key=out_time,
        )

        if self.hparams.dry_run:
            inp_files = inp_files[: 32 * (self.hparams.out_days + self.hparams.in_days)]

        # Checking for valid date format
        inp_invalid = lambda x: not (
            1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]) <= 31
        )
        assert not (sum([inp_invalid(x) for x in inp_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YYMMDD."
        )
        self.inp_files = sorted(inp_files, key=inp_time)

        out_invalid = lambda x: not (1 <= int(x[-5:-3]) <= 12)
        assert not (sum([out_invalid(x) for x in out_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YY_MM."
        )
        self.out_files = sorted(out_files, key=out_time)

        # Consider only ground truth and discard forecast values
        preprocess = lambda x: x.isel(time=slice(0, 1))

        with xr.open_mfdataset(
            # Remove duplicated file names
            set(inp_files),
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.input = ds.sortby("time").load()

        with xr.open_mfdataset(
            # Remove duplicated file names
            set(out_files),
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.output = ds.sortby("time").load()
            if self.hparams.round_to_zero:
                # Set values in range (0, `round_to_zero`) to small positive number
                self.output.frpfire.values[
                    (self.output.frpfire.values >= self.hparams.clip_output[0])
                    & (self.output.frpfire.values < self.hparams.clip_output[-1])
                ] = 1e-10
            if self.hparams.isolate_frp:
                # Setting isolated fire occurrence FRP to -1
                self.output.frpfire.values[
                    self.generate_isolated_mask(
                        # 1e-10 as a threshold since boxcox transformation accepts
                        # positive number only
                        self.output.frpfire.values
                        > 1e-10
                    )
                ] = -1

        # Ensure timestamp matches for both the input and output
        assert self.output.frpfire.time.min(skipna=True) <= self.input.rh.time.max(
            skipna=True
        )
        assert self.output.frpfire.time.max(skipna=True) >= self.input.rh.time.min(
            skipna=True
        )

        self.min_date = self.input.rh.time.min().values.astype("datetime64[D]")

        print(
            f"Start date: {self.output.frpfire.time.min(skipna=True)}",
            f"\nEnd date: {self.output.frpfire.time.max(skipna=True)}",
        )

        self.mask = torch.from_numpy(np.load(self.hparams.mask)).to(self.model.device)

    def generate_isolated_mask(self, x):
        """
        Generate the mask for value which have no fire occurrences for the day before \
and after.

        :param x: The numpy array to create the mask for
        :type x: ndarray
        :return: Mask for isolated values
        :rtype: ndarray
        """
        mask = x.copy()
        mask[0] = mask[0] & (x[0] | x[1])
        for i in range(1, x.shape[0] - 1):
            mask[i] = x[i] & (x[i - 1] | x[i + 1])
        mask[-1] = mask[-1] & (x[-1] | x[-2])
        return mask

    def training_step(self, model, batch):
        """
        Called inside the training loop with the data from the training dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """

        # forward pass
        x, y_pre = batch
        y_hat_pre = model(x)
        mask = model.data.mask.expand_as(y_pre[0][0])
        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_frp_to_zero:
                    y_hat = y_hat[y > self.hparams.round_frp_to_zero]
                    y = y[y > 0.5]
                if y_hat.nelement() == 0:
                    return {
                        "loss": torch.zeros(1, requires_grad=True),
                        "_log": None,
                    }
                y = y[y > 0.5]
                if self.hparams.boxcox:
                    y = torch.from_numpy(
                        stats.boxcox(
                            y.cpu()
                            if y.nelement() > 1
                            else np.concatenate([y.cpu(), y.cpu() + 1]),
                            lmbda=self.hparams.boxcox,
                        )
                    )[0 : y.shape[-1] if y.nelement() > 1 else 1].to(y_hat.device)
                pre_loss = (y_hat - y) ** 2
                loss = pre_loss.mean()
                assert loss == loss
                tensorboard_logs["train_loss_unscaled"][str(c)] = loss
        loss = torch.stack(
            list(tensorboard_logs["train_loss_unscaled"].values())
        ).mean()
        tensorboard_logs["_train_loss_unscaled"] = loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "loss": loss.true_divide(model.data.out_var * model.hparams.out_days),
            "_log": tensorboard_logs,
        }

    def validation_step(self, model, batch):
        """
        Called inside the validation loop with the data from the validation dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """

        # forward pass
        x, y_pre = batch
        y_hat_pre = model(x)
        mask = model.data.mask.expand_as(y_pre[0][0])
        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_frp_to_zero:
                    y_hat = y_hat[y > self.hparams.round_frp_to_zero]
                    y = y[y > 0.5]
                if y_hat.nelement() == 0:
                    return {}
                if self.hparams.boxcox:
                    y = torch.from_numpy(
                        stats.boxcox(
                            y.cpu()
                            if y.nelement() > 1
                            else np.concatenate([y.cpu(), y.cpu() + 1]),
                            lmbda=self.hparams.boxcox,
                        )
                    )[0 : y.shape[-1] if y.nelement() > 1 else 1].to(y.device)
                pre_loss = (y_hat - y) ** 2
                loss = pre_loss.mean()
                assert loss == loss

                # Accuracy for a threshold
                n_correct_pred = (
                    ((y - y_hat).abs() < model.hparams.thresh).float().mean()
                )
                abs_error = (y - y_hat).abs().float().mean()

                tensorboard_logs["val_loss"][str(c)] = loss
                tensorboard_logs["n_correct_pred"][str(c)] = n_correct_pred
                tensorboard_logs["abs_error"][str(c)] = abs_error

        val_loss = torch.stack(list(tensorboard_logs["val_loss"].values())).mean()
        tensorboard_logs["_val_loss"] = val_loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "val_loss": val_loss,
            "log": tensorboard_logs,
        }

    def test_step(self, model, batch):
        """
        Called inside the testing loop with the data from the testing dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """

        x, y_pre = batch
        y_hat_pre = model(x)
        mask = model.data.mask.expand_as(y_pre[0][0])
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_frp_to_zero:
                    y_hat = y_hat[y > self.hparams.round_frp_to_zero]
                    y = y[y > 0.5]
                if y_hat.nelement() == 0:
                    return {}
                if self.hparams.boxcox:
                    y_hat = torch.from_numpy(
                        inv_boxcox(y_hat.cpu().numpy(), self.hparams.boxcox)
                    ).to(y_hat.device)
                if self.hparams.clip_output:
                    y = y[
                        (y_hat < self.hparams.clip_output[-1])
                        & (self.hparams.clip_output[0] < y_hat)
                    ]
                    y_hat = y_hat[
                        (y_hat < self.hparams.clip_output[-1])
                        & (self.hparams.clip_output[0] < y_hat)
                    ]
                pre_loss = (
                    (y_hat - y).abs()
                    if model.hparams.loss == "mae"
                    else (y_hat - y) ** 2
                )
                loss = pre_loss.mean()
                assert loss == loss

                # Accuracy for a threshold
                n_correct_pred = (
                    ((y - y_hat).abs() < self.hparams.out_mad / 2).float().mean()
                )
                abs_error = (
                    (y - y_hat).abs().float().mean()
                    if model.hparams.loss == "mae"
                    else (y - y_hat).abs().float().mean()
                )

                tensorboard_logs["test_loss"][str(c)] = loss
                tensorboard_logs["n_correct_pred_test"][str(c)] = n_correct_pred
                tensorboard_logs["abs_error_test"][str(c)] = abs_error

        test_loss = torch.stack(list(tensorboard_logs["test_loss"].values())).mean()
        tensorboard_logs["_test_loss"] = test_loss

        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
        }
