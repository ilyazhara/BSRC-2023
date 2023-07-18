from typing import Optional

import pytorch_lightning as pl
import torch

from bnn_competition.tools.callbacks import ModelSaver


class Task:
    """
    This is a task for training.
    """

    def __init__(
        self,
        training_loop,
        epochs,
        model_saver,
        optimizers: Optional[dict] = None,
        check_val_every_n_epochs: Optional[int] = 50,
    ):
        """
        Args:
            training_loop: PL module which will be provided to PL trainer.
            epochs: Number of epochs to train.
            model_saver: Kwargs for saving checkpoint.
            optimizers (dict, optional): Dict of optimizers. Defaults to None.
            check_val_every_n_epochs (int, optional): The frequency (in epochs) of validation evaluating
            during training. Defaults to None.
            callbacks (list, optional): List of callbacks for PL training. Defaults to None.
        """
        self.training_loop = training_loop
        self.epochs = epochs
        self.model_saver = model_saver

        self.optimizers = optimizers
        self.check_val_every_n_epochs = check_val_every_n_epochs

    def set_optimizers(self):
        optimizers = []
        schedulers = []
        for _, opt in self.optimizers.items():
            optimizer_class = getattr(torch.optim, opt["name"])
            optimizer = optimizer_class(self.training_loop.model.parameters(), **opt["params"])

            scheduler_class = getattr(torch.optim.lr_scheduler, opt["scheduler"]["name"])
            scheduler = scheduler_class(optimizer, **opt["scheduler"]["params"])

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        self.training_loop.set_optimizers(optimizers=optimizers, schedulers=schedulers)

    def configure(self, logdir=None, num_gpus=1, checkpoint_path=None, **kwargs):
        self.set_optimizers()

        # The batch size comes from config where we expect the effective batch size,
        # whereas pytorch lightning expects the batch size per gpu.
        self.training_loop.dataset.batch_size //= num_gpus

        # logger
        logger = pl.loggers.TensorBoardLogger(logdir, name=None)

        # callbacks
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        model_saver = ModelSaver(logdir, **self.model_saver)
        progress_bar = pl.callbacks.RichProgressBar()

        # pl.Trainer
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=num_gpus,
            max_epochs=self.epochs,
            logger=logger,
            callbacks=[lr_monitor, model_saver, progress_bar],
            check_val_every_n_epoch=self.check_val_every_n_epochs,
        )
        self.checkpoint_path = checkpoint_path

    def run(self):
        self.trainer.fit(
            self.training_loop,
            self.training_loop.dataset.train_loader,
            self.training_loop.dataset.val_loader,
            ckpt_path=self.checkpoint_path,
        )
