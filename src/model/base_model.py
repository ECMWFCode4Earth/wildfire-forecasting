"""
Base model implementing helper methods.
"""
import pickle
from collections import defaultdict


import torch
from torch import optim
from torch.utils.data import DataLoader

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
        self.aux = False

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
        return self.data.test_step(self, batch)

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

        for n in range(self.data.n_output):
            tensorboard_logs[f"val_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["val_loss"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"val_acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["n_correct_pred"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["abs_error"] for x in outputs if x]]
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
        if outputs == [{}] * len(outputs):
            return {}
        avg_loss = torch.stack([x["test_loss"] for x in outputs if x]).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["test_loss"] = avg_loss

        for n in range(self.data.n_output):
            tensorboard_logs[f"test_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["test_loss"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"test_acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["n_correct_pred"] for x in outputs if x]]
            ).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["abs_error"] for x in outputs if x]]
            ).mean()

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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
            self.add_bias(self.data.out_mean)
            if self.hparams.test_set:
                if hasattr(self.hparams, "eval"):
                    self.train_data = self.test_data = self.data
                else:
                    self.train_data = self.data
                    hparams = self.hparams
                    hparams.eval = True
                    self.test_data = ModelDataset(
                        forecast_dir=self.hparams.forecast_dir,
                        forcings_dir=self.hparams.forcings_dir,
                        reanalysis_dir=self.hparams.reanalysis_dir,
                        frp_dir=self.hparams.frp_dir,
                        hparams=hparams,
                        out=self.hparams.out,
                    )
            else:
                self.train_data, self.test_data = torch.utils.data.random_split(
                    self.data,
                    [
                        len(self.data) * 8 // 10,
                        len(self.data) - len(self.data) * 8 // 10,
                    ],
                )
            if self.hparams.case_study and not self.hparams.test_set:
                assert (
                    max(self.test_data.indices) > 214
                ), "The data is outside the range of case study"
                self.test_data.indices = list(
                    set(self.test_data.indices) & set(range(214, 335))
                )

            # Saving list of test-set files
            if self.hparams.save_test_set:
                with open(self.hparams.save_test_set, "wb") as f:
                    pickle.dump(
                        [
                            self.test_data.indices,
                            sum(
                                [
                                    self.data.inp_files[i : i + 4]
                                    for i in self.test_data.indices
                                ],
                                [],
                            ),
                            [self.data.out_files[i] for i in self.test_data.indices],
                        ],
                        f,
                    )
            print()

            # Log test indices regardless
            log.info(self.test_data.indices)

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
