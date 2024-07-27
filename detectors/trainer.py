import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from pycocotools.coco import COCO
from torch import nn
from tqdm import tqdm

from detectors.data.coco_eval import CocoEvaluator
from detectors.utils import misc


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, output_path, device: torch.device = torch.device("cpu")):
        """Constructor for the Trainer class

        Args:
            output_path: Path to save the train outputs
            use_cuda: Whether to use the GPU
        """
        ## TODO: PROBALBY REMOVE THESE Initialize training objects
        # self.optimizer = optimizer_map[optimizer]
        # self.lr_scheduler = "test"

        self.device = device

        # Paths
        self.output_paths = {
            "output_dir": Path(
                f"{output_path}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
            ),
        }

    def train(
        self,
        model,
        criterion,
        dataloader_train,
        dataloader_val,
        val_coco_api: COCO,
        optimizer,
        scheduler,
        start_epoch=0,
        epochs=100,
        ckpt_every=None,
    ):
        """Train a model

        Args:
            model:
            optimizer:
            ckpt_every:
        """
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            train_stats = self._train_one_epoch(
                model, criterion, dataloader_train, optimizer
            )
            scheduler.step()

            self._evaluate(model, dataloader_val, val_coco_api)

            # Save the model every ckpt_every
            if ckpt_every is not None and (epoch + 1) % ckpt_every == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{epoch:04}"
                self._save_model(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_every,
                    save_path=ckpt_path,
                )

    def _train_one_epoch(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
    ):
        """Train one epoch

        Args:
            model: Model to train
            criterion: Loss function
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
            device: Device to train the model on
        """
        for steps, (samples, targets) in enumerate(dataloader_train):
            samples = samples.to(self.device)
            targets = [
                {key: value.to(self.device) for key, value in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()

            # len(bbox_predictions) = 3; bbox_predictions[i] (B, (5+n_class)*n_bboxes, out_w, out_h)
            bbox_predictions = model(samples)

            final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
                bbox_predictions, targets
            )

            # Calculate gradients and updates weights
            final_loss.backward()
            optimizer.step()
            break
        
    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        dataloader_val: Iterable,
        val_coco_api: COCO,
    ):
        """A single forward pass to evluate the val set after training an epoch

        Args:
            model: Model to train
            dataloader_val: Dataloader for the validation set
            device: Device to run the model on
        """

        model.eval()
        ## START HERE!!!!!!!!!!!!!! added val_coco_api hopefully it's the right one

        coco_evaluator = CocoEvaluator(
            val_coco_api, iou_types=["bbox"]
        )  # , bbox_fmt="coco")

        for steps, (samples, targets) in enumerate(dataloader_val):
            samples = samples.to(self.device)
            targets = [
                {key: value.to(self.device) for key, value in t.items()}
                for t in targets
            ]

            bbox_predictions = model(samples, inference=True)

            # for image, target, boxes, confs in zip(images, targets,)
            ### START HERE
            # final_loss.backward()

    def _save_model(
        self, model, optimizer, lr_scheduler, current_epoch, ckpt_every, save_path
    ):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": current_epoch,
            },
            save_path,
        )
