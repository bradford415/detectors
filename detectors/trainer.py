import datetime
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from pycocotools.coco import COCO
from torch import nn
from tqdm import tqdm

from detectors.data.coco_eval import CocoEvaluator
from detectors.data.coco_utils import convert_to_coco_api
from detectors.utils import misc
from detectors.utils.box_ops import val_preds_to_img_size


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, output_path: str, device: torch.device = torch.device("cpu"), logging_intervals: Dict = {}):
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
            "output_dir": Path(output_path),
        }
        
        self.log_intervals = logging_intervals
        if not logging_intervals:
            self.log_intervals = {"train_steps_freq": 100}

    def train(
        self,
        model,
        criterion,
        dataloader_train,
        dataloader_val,
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
        print("\nStart training")
        start_time = time.time()

        # Starting the epoch at 1 makes calculations more intuitive
        start_epoch = 1  # Once checkpointing is implemented we can overwrite this value
        for epoch in range(start_epoch, epochs):

            ## TODO: Implement tensorboard as shown here: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/logger.py#L6
            train_stats = self._train_one_epoch(
                model, criterion, dataloader_train, optimizer
            )
            scheduler.step()

            test_stats, coco_evaluator = self._evaluate(
                model, criterion, dataloader_val
            )

            # Save the model every ckpt_every
            if ckpt_every is not None and (epoch) % ckpt_every == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{epoch:04}"
                self._save_model(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_every,
                    save_path=ckpt_path,
                )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))

            bbox_stats = coco_evaluator.coco_eval["bbox"].stats
            print(bbox_stats)

    def _train_one_epoch(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ):
        """Train one epoch

        Args:
            model: Model to train
            criterion: Loss function
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
            epoch: Used for logging purposes
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

            if steps % self.log_intervals["train_steps_freq"] == 0:
                print(
                    f"epoch: {epoch}\t''iter: {steps}/{len(dataloader_train)}\t''loss: {final_loss:.4f}")

            # Calculate gradients and updates weights
            final_loss.backward()
            optimizer.step()
            break

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_val: Iterable,
    ):
        """A single forward pass to evluate the val set after training an epoch

        Args:
            model: Model to train
            criterion: Loss function; only used to inspect the loss on the val set,
                       not used for backpropagation
            dataloader_val: Dataloader for the validation set
            device: Device to run the model on
        """

        model.eval()
        ########################## START HERE - I THINK I NEED TO GRAB THE COCO API FROM THIS METHOD
        # https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

        val_coco_api = convert_to_coco_api(dataloader_val.dataset, bbox_fmt="coco")
        coco_evaluator = CocoEvaluator(
            val_coco_api, iou_types=["bbox"], bbox_format="coco"
        )

        for steps, (samples, targets) in enumerate(dataloader_val):
            samples = samples.to(self.device)
            targets = [
                {key: value.to(self.device) for key, value in t.items()}
                for t in targets
            ]

            # Inference outputs bbox_preds (cx, cy, w, h) and class confidences (num_classes);
            # TODO: These should all be between 0-1 but some look greater than 1, need to investigate
            bbox_preds, class_conf = model(samples, inference=True)

            # final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
            #    bbox_predictions, targets
            # )

            ## TODO: Comment this
            results = val_preds_to_img_size(samples, targets, bbox_preds, class_conf)

            # TODO: placeholder; change later
            test_stats = 5

            ## TODO: still VERY fuzzy on what the network actually predicts during validation and
            #        how we scale back to original image size
            # breakpoint()
            ### START HERE
            evaluator_time = time.time()
            coco_evaluator.update(results)
            evaluator_time = time.time() - evaluator_time

        coco_evaluator.synchronize_between_processes()

        # Accumulate predictions from all processes
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        ################### START HERE continue with val loop ####################
        return test_stats, coco_evaluator

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
