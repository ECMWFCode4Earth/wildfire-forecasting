"""
Base Dataset class to work with fwi-forcings data.
"""
from collections import defaultdict
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from imblearn.under_sampling import RandomUnderSampler


class ModelDataset(Dataset):
    """
    The dataset class responsible for loading the data and providing the samples for \
training.

    :param Dataset: Base Dataset class to use with PyTorch models
    :type Dataset: torch.utils.data.Dataset
    """

    def __init__(
        self,
        out_var=None,
        out_mean=None,
        forecast_dir=None,
        forcings_dir=None,
        reanalysis_dir=None,
        transform=None,
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
defaults to None
        :type reanalysis_dir: str, optional
        :param transform: Custom transform for the input variable, defaults to None
        :type transform: torch.transforms, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        self.hparams = hparams
        self.out_mean = out_mean
        self.out_var = out_var
        self.hparams.thresh = self.hparams.out_mad / 2
        if self.hparams.binned:
            self.bin_intervals = self.hparams.binned

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else self.hparams.out_mean
        # Variance of output variable used to scale the training loss.
        self.out_var = out_var if out_var else self.hparams.out_var

        # Convert string dates to numpy format
        if self.hparams.date_range:
            self.hparams.date_range = [
                np.datetime64(d) for d in self.hparams.date_range
            ]
        # Convert case-study dates to numpy format
        if (
            hasattr(self.hparams, "case_study_dates")
            and self.hparams.case_study_dates
            and not self.hparams.date_range
        ):
            self.hparams.case_study_dates = [
                [np.datetime64(d) for d in r] for r in self.hparams.case_study_dates
            ]
        # If custom date range specified, override
        else:
            self.hparams.case_study_dates = None

        # Create imbalanced-learn random subsampler
        if self.hparams.undersample:
            self.undersampler = RandomUnderSampler()

        if not self.hparams.benchmark:
            # Input transforms including mean and std normalization
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Mean and standard deviation stats used to normalize the input data
                    # to the mean of zero and standard deviation of one.
                    transforms.Normalize(
                        [
                            x
                            for i in range(self.hparams.in_days)
                            for x in (
                                self.hparams.inp_mean["rh"],
                                self.hparams.inp_mean["t2"],
                                self.hparams.inp_mean["tp"],
                                self.hparams.inp_mean["wspeed"],
                                self.hparams.inp_mean["skt"]
                            )
                        ]
                        + (
                            [
                                self.hparams.smos_mean
                                for i in range(self.hparams.in_days)
                            ]
                            if self.hparams.smos_input
                            else []
                        ),
                        [
                            x
                            for i in range(self.hparams.in_days)
                            for x in (
                                self.hparams.inp_std["rh"],
                                self.hparams.inp_std["t2"],
                                self.hparams.inp_std["tp"],
                                self.hparams.inp_std["wspeed"],
                                self.hparams.inp_std["skt"],
                            )
                        ]
                        + (
                            [self.hparams.smos_std for i in range(self.hparams.in_days)]
                            if self.hparams.smos_input
                            else []
                        ),
                    ),
                ]
            )

    def __len__(self):
        """
        The internal method used to obtain the number of iteration samples.

        :return: The maximum possible iterations with the provided data.
        :rtype: int
        """
        return len(self.dates)

    def __getitem__(self, idx):
        """
        Internal method used by pytorch to fetch input and corresponding output tensors.

        :param idx: The index number of data sample.
        :type idx: int
        :return: Batch of data containing input and output tensors
        :rtype: tuple
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.hparams.benchmark:
            X = torch.from_numpy(
                np.stack(
                    [
                        resize(
                            self.input[list(self.input.data_vars)[0]]
                            .sel(time=[self.dates[idx]], lead=[i])
                            .values.squeeze(),
                            self.output[list(self.output.data_vars)[0]][0].shape,
                        )
                        for i in range(self.hparams.out_days)
                    ],
                    axis=0,
                )
            )
        else:
            X = self.transform(
                np.stack(
                    [
                        self.input[v]
                        .sel(time=[self.dates[idx] - np.timedelta64(i, "D")])
                        .values.squeeze()
                        for i in range(self.hparams.in_days)
                        for v in ["rh", "t2", "tp", "wspeed", "skt"]
                    ]
                    + (
                        [
                            resize(
                                np.nan_to_num(
                                    self.smos_input[list(self.smos_input.data_vars)[0]]
                                    .sel(
                                        time=[self.dates[idx] - np.timedelta64(i, "D")],
                                        method="nearest",
                                    )
                                    .values.squeeze()[::-1],
                                    copy=False,
                                    # Use 50 as the placeholder for water bodies
                                    nan=50,
                                ),
                                self.input.rh[0].shape,
                            )
                            for i in range(self.hparams.in_days)
                        ]
                        if self.hparams.smos_input
                        else []
                    ),
                    axis=-1,
                )
            )

        y = torch.from_numpy(
            np.stack(
                [
                    self.output[list(self.output.data_vars)[0]]
                    .sel(time=[self.dates[idx] + np.timedelta64(i, "D")])
                    .values.squeeze()
                    for i in range(self.hparams.out_days)
                ],
                axis=0,
            )
        )

        return X, y

    def get_cb_loss_factor(self, y):
        """
        Compute the Class-Balanced loss factor mask using output value frequency \
distribution and the supplied beta factor.

        :param y: The 1D ground truth value tensor
        :type y: torch.tensor
        """
        idx = (
            (
                y.unsqueeze(0).expand(self.bin_centers.shape[0], -1)
                - self.bin_centers.unsqueeze(-1).expand(-1, y.shape[0])
            )
            .abs()
            .argmin(dim=0)
        )
        loss_factor = torch.empty_like(y)
        for i in range(self.bin_centers.shape[0]):
            loss_factor[idx == i] = self.loss_factors[i]
        return loss_factor

    def apply_mask(self, *y_list):
        """
        Returns batch_size x channels x N sized matrices after applying the mask.

        :param *y_list: The interable of tensors to be masked
        :type y_list: torch.Tensor
        :return: The list of masked tensors
        :rtype: list(torch.Tensor)
        """
        return [
            y.permute(-2, -1, 0, 1)[self.mask.expand_as(y[0][0])].permute(-2, -1, 0)
            for y in y_list
        ]

    def get_loss(self, y, y_hat):
        """
        Do the applicable processing and return the loss for the supplied prediction \
and the label tensors.

        :param y: Label tensor
        :type y: torch.Tensor
        :param y_hat: Predicted tensor
        :type y_hat: torch.Tensor
        :return: Prediction loss
        :rtype: torch.Tensor
        """
        if self.hparams.undersample:
            sub_mask = y < self.hparams.undersample
            subval = y[sub_mask]
            low = max(subval.min(), 0.5)
            high = subval.max()
            boundaries = torch.arange(low, high, (high - low) / 10).to(
                self.model.device
            )
            freq_idx = torch.bucketize(subval, boundaries[:-1], right=False)
            self.undersampler.fit_resample(
                subval.cpu().unsqueeze(-1),
                (boundaries.take(index=freq_idx).cpu() * 100).int(),
            )
            idx = self.undersampler.sample_indices_
            y = torch.cat((y[~sub_mask], subval[idx]))
            y_hat = torch.cat((y_hat[~sub_mask], y_hat[sub_mask][idx]))

        if self.hparams.round_to_zero:
            y_hat = y_hat[y > self.hparams.round_to_zero]
            y = y[y > self.hparams.round_to_zero]

        if self.hparams.clip_output:
            y_hat = y_hat[
                (y < self.hparams.clip_output[-1]) & (self.hparams.clip_output[0] < y)
            ]
            y = y[
                (y < self.hparams.clip_output[-1]) & (self.hparams.clip_output[0] < y)
            ]

        if self.hparams.cb_loss:
            loss_factor = self.get_cb_loss_factor(y)

        if self.hparams.boxcox:
            y = torch.from_numpy(boxcox(y.cpu(), lmbda=self.hparams.boxcox,)).to(
                y.device
            )

        pre_loss = (y_hat - y) ** 2
        # if "loss_factor" in locals():
        #     pre_loss *= loss_factor
        loss = pre_loss.mean()
        assert loss == loss

        return loss

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
        y_pre, y_hat_pre = self.apply_mask(y_pre, y_hat_pre)

        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                loss = self.get_loss(y_pre[b][c], y_hat_pre[b][c])

                tensorboard_logs["train_loss_unscaled"][str(c)] = loss
        loss = torch.stack(
            list(tensorboard_logs["train_loss_unscaled"].values())
        ).mean()
        tensorboard_logs["_train_loss_unscaled"] = loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "loss": loss.true_divide(model.data.out_var * self.hparams.out_days),
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
        y_pre, y_hat_pre = self.apply_mask(y_pre, y_hat_pre)

        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y, y_hat = y_pre[b][c], y_hat_pre[b][c]
                loss = self.get_loss(y, y_hat)

                # Accuracy for a threshold
                abs_diff = (y - y_hat).abs()
                acc = (abs_diff < self.hparams.thresh).float().mean()
                mae = abs_diff.mean()

                tensorboard_logs["val_loss"][str(c)] = loss
                tensorboard_logs["acc"][str(c)] = acc
                tensorboard_logs["mae"][str(c)] = mae

        val_loss = torch.stack(list(tensorboard_logs["val_loss"].values())).mean()
        tensorboard_logs["_val_loss"] = val_loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "val_loss": val_loss,
            "log": tensorboard_logs,
        }

    def inference_step(self, y_pre, y_hat_pre):
        """
        Run inference for the target and predicted values and return the loss and the \
metrics values as logs.

        :param y_pre: Label values
        :type y_pre: torch.Tensor
        :param y_hat_pre: Predicted value
        :type y_hat_pre: torch.Tensor
        :return: Loss and the log dictionary
        :rtype: tuple
        """
        y_pre, y_hat_pre = self.apply_mask(y_pre, y_hat_pre)

        tensorboard_logs = defaultdict(dict)

        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c]
                y_hat = y_hat_pre[b][c]

                if self.hparams.boxcox and not self.hparams.benchmark:
                    # Negative predictions give NaN after inverse-boxcox
                    y_hat[y_hat < 0] = 0
                    y_hat = torch.from_numpy(
                        inv_boxcox(y_hat.cpu().numpy(), self.hparams.boxcox)
                    ).to(y_hat.device)

                if not y.numel():
                    return None

                pre_loss = (y_hat - y) ** 2

                loss = lambda low, high: pre_loss[(y > low) & (y <= high)].mean()
                assert loss(y.min(), y.max()) == loss(y.min(), y.max())

                # Accuracy for a threshold
                acc = (
                    lambda low, high: (
                        (y - y_hat)[(y > low) & (y <= high)].abs() < self.hparams.thresh
                    )
                    .float()
                    .mean()
                )

                # Mean absolute error
                mae = (
                    lambda low, high: (y - y_hat)[(y > low) & (y <= high)]
                    .abs()
                    .float()
                    .mean()
                )

                tensorboard_logs["mse"][str(c)] = loss(y.min(), y.max())
                tensorboard_logs["acc"][str(c)] = acc(y.min(), y.max())
                tensorboard_logs["mae"][str(c)] = mae(y.min(), y.max())

                # Inference on binned values
                if self.hparams.binned:
                    for i in range(len(self.bin_intervals) - 1):
                        low, high = (
                            self.bin_intervals[i],
                            self.bin_intervals[i + 1],
                        )
                        tensorboard_logs[f"mse_{low}_{high}"][str(c)] = loss(low, high)
                        tensorboard_logs[f"acc_{low}_{high}"][str(c)] = acc(low, high)
                        tensorboard_logs[f"mae_{low}_{high}"][str(c)] = mae(low, high)
                    tensorboard_logs[f"mse_{self.bin_intervals[-1]}inf"][str(c)] = loss(
                        self.bin_intervals[-1], y.max()
                    )
                    tensorboard_logs[f"acc_{self.bin_intervals[-1]}inf"][str(c)] = acc(
                        self.bin_intervals[-1], y.max()
                    )
                    tensorboard_logs[f"mae_{self.bin_intervals[-1]}inf"][str(c)] = mae(
                        self.bin_intervals[-1], y.max()
                    )

        inference_loss = torch.stack(list(tensorboard_logs["mse"].values())).mean()
        tensorboard_logs["_inference_loss"] = inference_loss

        return inference_loss, tensorboard_logs

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

        test_loss, tensorboard_logs = self.inference_step(y_pre, y_hat_pre)

        return {
            "mse": test_loss,
            "log": tensorboard_logs,
        }

    def benchmark_step(self, batch):
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
        y_hat_pre, y_pre = batch

        benchmark_loss, tensorboard_logs = self.inference_step(y_pre, y_hat_pre)

        return {
            "mse": benchmark_loss,
            "log": tensorboard_logs,
        }
