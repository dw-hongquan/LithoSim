'''
Author: Hongquan He
Date: Apr 07, 2025
'''
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
import torchvision.utils as U
import torchvision.transforms as T
from datetime import datetime
import os
import aim
from pathlib import Path
from lightning import LightningModule
from aim.sdk.adapters.pytorch_lightning import AimLogger
from src.models.losses.metrics import iou_dice
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy


class LithoLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        activate: bool = True,
        visual_in_val: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.save_dir = Path(os.path.join('./data/litho/', datetime.now().strftime("%y%m%d_%H%M%S")))

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        return self.net(source, mask, dose, defocus)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()

    def model_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        source, mask, dose, defocus, resist = batch
        resist_pred = self.forward(source, mask, dose, defocus)
        loss = self.criterion(resist_pred, resist)
        return loss, resist_pred, resist

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, _, _ = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        loss, y_pred, y = self.model_step(batch)
        iou, dice = iou_dice(y_pred, y, activate=self.hparams.activate)

        self.val_loss(loss)
        self.val_acc(y_pred, y)
        self.log("val/loss", self.val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/dice", dice, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.hparams.visual_in_val:
            validation_dir = self.save_dir / f'validation'
            validation_dir.mkdir(parents = True, exist_ok = True)
            if self.global_rank == 0:
                if isinstance(self.logger, AimLogger):
                    transform  = T.ToPILImage()
                    grid_pred = U.make_grid(y_pred, nrow = 4)
                    grid_gt = U.make_grid(y, nrow = 4)
                    pred_val_path = validation_dir / (str(self.current_epoch) + '_' + str(batch_idx) + '_pred.png')
                    gt_val_path = validation_dir / (str(self.current_epoch) + '_' + str(batch_idx) + '_gt.png')
                    U.save_image(grid_pred, pred_val_path)
                    U.save_image(grid_gt, gt_val_path)
                    aim_images = [
                        aim.Image(transform(ii))
                        for ii in [
                            grid_pred.detach().clone(),
                            grid_gt.detach().clone(),
                        ]
                    ]
                    self.logger.experiment.track(
                        value=aim_images,
                        name=f"pred in epoch {self.current_epoch}",
                        step=self.global_step,
                        context={"epoch": self.current_epoch},
                    )


    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        loss, y_pred, y = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(y_pred, y)

        iou, dice = iou_dice(y_pred, y, activate=self.hparams.activate)

        self.log("test/loss", self.test_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.test_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test/iou", iou, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test/dice", dice, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        grid_pred = U.make_grid(y_pred, nrow = 4)
        grid_gt = U.make_grid(y, nrow = 4)

        test_dir = self.save_dir / f'test'
        test_dir.mkdir(parents = True, exist_ok = True)
        pred_test_path = test_dir / (str(batch_idx) + '_pred.png')
        gt_test_path = test_dir / (str(batch_idx) + '_gt.png')
        U.save_image(grid_pred, pred_test_path)
        U.save_image(grid_gt, gt_test_path)

    def on_test_epoch_end(self) -> None:

        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss", # 'val/loss' is not available.
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LithoLitModule(None, None, None, None, None, None)
