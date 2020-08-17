"""
Base model implementing helper methods.
"""
from collections import defaultdict

import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from skimage.transform import resize

# Logging helpers
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


class BaseModel(LightningModule):
    """
    The primary class containing all the training functionality. It is equivalent to\
PyTorch nn.Module in all aspects.

    :param LightningModule: The Pytorch-Lightning module derived from nn.module with\
useful hooks
    :type LightningModule: nn.Module
    :raises NotImplementedError: Some methods must be overridden
    """

    def __init__(self, hparams):
        """
        Constructor for BaseModel.

        :param hparams: Holds configuration values
        :type hparams: Namespace
        """

        # init superclass
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.data_prepared = False

    def forward(self):
        """
        Dummy method to do forward pass on the model.

        :raises NotImplementedError: The method must be overridden in the derived models
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        Called inside the testing loop with the data from the testing dataloader \
passed in as `batch`. The implementation is delegated to the dataloader instead.

        For performance critical usecase prefer monkey-patching instead.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """
        return self.data.training_step(self, batch)

    def validation_step(self, batch, batch_idx):
        """
        Called inside the validation loop with the data from the validation dataloader \
passed in as `batch`. The implementation is delegated to the dataloader instead.

        For performance critical usecase prefer monkey-patching instead.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """
        return self.data.validation_step(self, batch)

    def test_step(self, batch, batch_idx):
        """
        Called inside the testing loop with the data from the testing dataloader \
passed in as `batch`. The implementation is delegated to the dataloader instead.

        For performance critical usecase prefer monkey-patching instead.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """
        return (
            self.data.benchmark_step(batch)
            if self.hparams.benchmark
            else self.data.test_step(self, batch)
        )

    def training_epoch_end(self, outputs):
        """
        Called at the end of training epoch to aggregate outputs.

        :param outputs: List of individual outputs of each training step.
        :type outputs: list
        :return: Loss and logs.
        :rtype: dict
        """
        if outputs == [{}] * len(outputs):
            return {"loss": torch.zeros(1, requires_grad=True)}
        avg_loss = torch.stack(
            [x["_log"]["_train_loss_unscaled"] for x in outputs if x["_log"]]
        ).mean()
        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["train_loss"] = avg_loss
        return {
            "train_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        :param outputs: List of individual outputs of each validation step.
        :type outputs: list
        :return: Loss and logs.
        :rtype: dict
        """
        if outputs == [{}] * len(outputs):
            return {}
        avg_loss = torch.stack([x["val_loss"] for x in outputs if x]).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["val_loss"] = avg_loss

        for n in range(self.hparams.out_days):
            tensorboard_logs[f"val_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["val_loss"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"val_acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["acc"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"mae_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["mae"] for x in outputs if x]]
            ).mean()

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        """
        Called at the end of testing epoch to aggregate outputs.

        :param outputs: List of individual outputs of each testing step.
        :type outputs: list
        :return: Loss and logs.
        :rtype: dict
        """

        ifx = lambda x: x if x else [torch.zeros(1)]
        rm_none = lambda x: ifx([t for t in x if not torch.isnan(t).any()])
        avg_loss = torch.stack(rm_none([x["mse"] for x in outputs])).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["mse"] = avg_loss

        for n in range(self.hparams.out_days):
            tensorboard_logs[f"mse_{n}"] = torch.stack(
                rm_none([d[str(n)] for d in [x["log"]["mse"] for x in outputs]])
            ).mean()
            tensorboard_logs[f"acc_{n}"] = torch.stack(
                rm_none([d[str(n)] for d in [x["log"]["acc"] for x in outputs]])
            ).mean()
            tensorboard_logs[f"mae_{n}"] = torch.stack(
                rm_none([d[str(n)] for d in [x["log"]["mae"] for x in outputs]])
            ).mean()

            # Inference on binned values
            if self.hparams.binned:
                for i in range(len(self.data.bin_intervals) - 1):
                    low, high = (
                        self.data.bin_intervals[i],
                        self.data.bin_intervals[i + 1],
                    )
                    tensorboard_logs[f"mse_{low}_{high}_{n}"] = torch.stack(
                        rm_none(
                            [
                                d[str(n)]
                                for d in [
                                    x["log"][f"mse_{low}_{high}"] for x in outputs
                                ]
                            ]
                        )
                    ).mean()
                    tensorboard_logs[f"acc_{low}_{high}_{n}"] = torch.stack(
                        rm_none(
                            [
                                d[str(n)]
                                for d in [
                                    x["log"][f"acc_{low}_{high}"] for x in outputs
                                ]
                            ]
                        )
                    ).mean()
                    tensorboard_logs[f"mae_{low}_{high}_{n}"] = torch.stack(
                        rm_none(
                            [
                                d[str(n)]
                                for d in [
                                    x["log"][f"mae_{low}_{high}"] for x in outputs
                                ]
                            ]
                        )
                    ).mean()
                tensorboard_logs[
                    f"mse_{self.data.bin_intervals[-1]}_inf_{n}"
                ] = torch.stack(
                    rm_none(
                        [
                            d[str(n)]
                            for d in [
                                x["log"][f"mse_{self.data.bin_intervals[-1]}inf"]
                                for x in outputs
                            ]
                        ]
                    )
                ).mean()
                tensorboard_logs[
                    f"acc_{self.data.bin_intervals[-1]}_inf_{n}"
                ] = torch.stack(
                    rm_none(
                        [
                            d[str(n)]
                            for d in [
                                x["log"][f"acc_{self.data.bin_intervals[-1]}inf"]
                                for x in outputs
                            ]
                        ]
                    )
                ).mean()
                tensorboard_logs[
                    f"mae_{self.data.bin_intervals[-1]}_inf_{n}"
                ] = torch.stack(
                    rm_none(
                        [
                            d[str(n)]
                            for d in [
                                x["log"][f"mae_{self.data.bin_intervals[-1]}inf"]
                                for x in outputs
                            ]
                        ]
                    )
                ).mean()

        try:
            self.logger.experiment[0].log(tensorboard_logs)
        except:
            log.info("Logger not found, skipping the log step.")
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Decide optimizers and learning rate schedulers.

        At least one optimizer is required.

        :return: Optimizer and the schedular
        :rtype: tuple
        """
        if self.hparams.benchmark:
            return None

        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate,)
        if self.hparams.optim == "cosine":
            scheduler = [
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=0, verbose=True, threshold=1e-1
                ),
            ]
        elif self.hparams.optim == "one_cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=len(self.train_data),
                epochs=self.hparams.epochs,
            )
        return [optimizer], [scheduler]

    def add_bias(self, bias):
        """
        Initialize bias parameter of the last layer with the output variable's mean.

        :param bias: Mean of the output variable.
        :type bias: float
        """
        for w in reversed(self.state_dict().keys()):
            if "bias" in w:
                self.state_dict()[w].fill_(bias)
                break

    def prepare_data(self, ModelDataset=None, force=False):
        """
        Load and split the data for training and test during the first call. Behavior \
on second call determined by the `force` parameter.

        :param ModelDataset: The dataset class to be used with the model, defaults to
            None
        :type ModelDataset: class, optional
        :param force: Force the data preperation even if already prepared, defaults to
            False
        :type force: bool, optional
        """
        if self.data_prepared and not force:
            pass
        elif ModelDataset:
            self.data = ModelDataset(
                forecast_dir=self.hparams.forecast_dir,
                forcings_dir=self.hparams.forcings_dir,
                reanalysis_dir=self.hparams.reanalysis_dir,
                frp_dir=self.hparams.frp_dir,
                hparams=self.hparams,
                out=self.hparams.out,
            )
            self.data.model = self
            if self.hparams.cb_loss:
                # Move bin_centers and freq to GPU if possible
                self.data.bin_centers = torch.from_numpy(self.hparams.bin_centers).to(
                    self.device, dtype=next(iter(self.data))[1].dtype
                )
                self.data.loss_factors = torch.from_numpy(self.hparams.loss_factors).to(
                    self.device, dtype=next(iter(self.data))[1].dtype
                )

            if self.hparams.smos_input:
                self.data.mask[0:105, :] = False

            if self.hparams.benchmark:
                self.data.input = self.data.BenchmarkDataset(
                    dates=self.data.dates,
                    forecast_dir=self.hparams.forecast_dir,
                    hparams=self.hparams,
                ).output

            # Load the mask for output variable if provided or generate from NaN mask
            nan_mask = ~np.isnan(
                self.data.output[list(self.data.output.data_vars)[0]][0].values
            )
            if self.hparams.benchmark:
                nan_mask &= ~np.isnan(
                    resize(
                        self.data.input[list(self.data.input.data_vars)[0]][0][
                            0
                        ].values,
                        self.data.output[list(self.data.output.data_vars)[0]][0].shape,
                    )
                )
            if self.hparams.mask:
                nan_mask &= np.load(self.hparams.mask)
            self.data.mask = torch.from_numpy(nan_mask).to(self.device)

            self.add_bias(self.data.out_mean)

            if not hasattr(self.hparams, "eval"):
                if self.hparams.chronological_split:
                    self.train_data = torch.utils.data.Subset(
                        self.data,
                        range(int(len(self.data) * self.hparams.chronological_split)),
                    )
                    self.test_data = torch.utils.data.Subset(
                        self.data,
                        range(
                            int(len(self.data) * self.hparams.chronological_split),
                            len(self.data),
                        ),
                    )
                else:
                    self.train_data, self.test_data = torch.utils.data.random_split(
                        self.data,
                        [
                            len(self.data) * (5 if self.hparams.dry_run else 8) // 10,
                            len(self.data)
                            - len(self.data) * (5 if self.hparams.dry_run else 8) // 10,
                        ],
                    )
            else:
                self.train_data = self.test_data = self.data
                self.test_data.indices = list(range(len(self.test_data)))

            test_set_dates = [
                str(self.data.min_date + np.timedelta64(i, "D"))
                for i in self.test_data.indices
            ]
            log.info(test_set_dates)

            # Set flag to avoid resource intensive re-preparation during next call
            self.data_prepared = True

    def train_dataloader(self):
        """
        Create the training dataloader from the training dataset.

        :return: The training dataloader
        :rtype: Dataloader
        """
        log.info("Training data loader called.")
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            shuffle=True,
            pin_memory=True if self.hparams.gpus else False,
        )

    def val_dataloader(self):
        """
        Create the validation dataloader from the validation dataset.

        :return: The validation dataloader
        :rtype: Dataloader
        """
        log.info("Validation data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            pin_memory=True if self.hparams.gpus else False,
        )

    def test_dataloader(self):
        """
        Create the testing dataloader from the testing dataset.

        :return: The testing dataloader
        :rtype: Dataloader
        """
        log.info("Test data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            pin_memory=True if self.hparams.gpus else False,
        )
