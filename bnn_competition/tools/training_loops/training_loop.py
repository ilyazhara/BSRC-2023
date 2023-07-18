from typing import Dict, Optional

import pytorch_lightning as pl
import torch


class TrainingLoop(pl.LightningModule):
    """
    Class containing training pipeline.
    """

    def __init__(
        self,
        dataset,
        model,
        loss,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            dataset: Dataset for training and evaluation.
            model: The model to train.
            loss: Target loss.
            train_metrics (dict, optional): Dict of metrics for train. Defaults to None.
            val_metrics (dict, optional): Dict of metrics for validation. Defaults to None.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.model = model
        self.loss = loss

        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}
        self.train_metrics_names = list(train_metrics.keys())
        self.val_metrics_names = list(val_metrics.keys())
        for name, metric in train_metrics.items():
            setattr(self, "train_" + name, metric)
        for name, metric in val_metrics.items():
            setattr(self, "val_" + name, metric)

    @property
    def automatic_optimization(self):
        return False

    def set_optimizers(self, optimizers, schedulers):
        self.optimizer = optimizers
        self.scheduler = schedulers

    def configure_optimizers(self):
        return self.optimizer, self.scheduler

    def log_losses(self, prefix="", progress_bar=True):
        for name, value in self.logs.items():
            self.log(prefix + name, value, on_step=False, on_epoch=True, sync_dist=True)
            if progress_bar:
                self.log("pb_" + prefix + name, value, logger=False, on_step=True, prog_bar=True, sync_dist=True)

    def log_metrics(self, predictions, targets, prefix="", progress_bar=True):
        for name in getattr(self, prefix + "metrics_names"):
            getattr(self, prefix + name).update(predictions, targets)

        if progress_bar:
            for name in getattr(self, prefix + "metrics_names"):
                self.log(
                    "pb_" + prefix + name,
                    getattr(self, prefix + name).compute(),
                    logger=False,
                    on_step=True,
                    prog_bar=True,
                    sync_dist=True,
                )

    def compute_total_loss(self, model_outputs, batch):
        x, targets = batch

        loss = self.loss(model_outputs, targets)

        self.logs["target_loss"] = torch.clone(loss)
        return loss

    def optimization_step(self, loss):
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

    def get_model_inputs(self, batch):
        return batch[0]

    def get_targets(self, batch):
        return batch[1]

    def training_step(self, train_batch, batch_idx, logs=None):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(train_batch))

        loss = self.compute_total_loss(model_outputs, train_batch)
        self.optimization_step(loss)
        self.log_losses(prefix="train_")
        self.log_metrics(model_outputs, self.get_targets(train_batch), prefix="train_")

        return loss

    def training_epoch_end(self, outs):
        for name in self.train_metrics_names:
            self.log("train_" + name, getattr(self, "train_" + name).compute(), sync_dist=True)
            getattr(self, "train_" + name).reset()

    def validation_step(self, val_batch, batch_idx):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(val_batch))

        loss = self.compute_total_loss(model_outputs, val_batch)
        self.log_losses(prefix="val_", progress_bar=False)
        self.log_metrics(model_outputs, self.get_targets(val_batch), prefix="val_", progress_bar=False)

        return loss

    def validation_epoch_end(self, outputs):
        for name in self.val_metrics_names:
            self.log("val_" + name, getattr(self, "val_" + name).compute(), sync_dist=True)
            getattr(self, "val_" + name).reset()
