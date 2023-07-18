import os

import pytorch_lightning as pl
import torch


class ModelSaver(pl.callbacks.ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        torch.save(pl_module.model, os.path.join(trainer.logger.log_dir, "model.ckpt"))
