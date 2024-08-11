import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.utils import data
from tqdm import tqdm

from detectors.data.coco_eval import CocoEvaluator
from detectors.data.coco_utils import convert_to_coco_api
from detectors.utils import misc
from detectors.utils.box_ops import val_preds_to_img_size

log = logging.getLogger(__name__)


class Trainer:
    """Trainer TODO: comment"""

    def __init__(
        self,
        output_path: str,
        device: torch.device = torch.device("cpu"),
        logging_intervals: Dict = {},
    ):
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
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: data.DataLoader,
        dataloader_val: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        start_epoch: int = 1,
        epochs: int = 100,
        ckpt_every: int = None,
    ):
        """Trains a model

        Specifically, this method trains a model for n epochs and evaluates on the validation set.
        A model checkpoint is saved at user-specified intervals

        Args:
            model: A pytorch model to be trained
            criterion: The loss function to use for training
            dataloader_train: Torch dataloader to loop through the train dataset
            dataloader_val: Torch dataloader to loop through the val dataset
            optimizer: Optimizer which determines how to update the weights
            scheduler: Scheduler which determines how to change the learning rate
            start_epoch: Epoch to start the training on; starting at 1 is a good default because it makes
                         checkpointing and calculations more intuitive
            epochs: The epoch to end training on; unless starting from a check point, this will be the number of epochs to train for
            ckpt_every: Save the model after n epochs
        """
        log.info("\nTraining started\n")
        total_train_start_time = time.time()

        # Starting the epoch at 1 makes calculations more intuitive
        for epoch in range(start_epoch, epochs):
            ## TODO: Implement tensorboard as shown here: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/logger.py#L6

            # Track the time it takes for one epoch (train and val)
            one_epoch_start_time = time.time()

            # Train one epoch
            train_stats = self._train_one_epoch(
                model, criterion, dataloader_train, optimizer, epoch
            )
            scheduler.step()

            # Evaluate the model on the validation set
            coco_evaluator = self._evaluate(model, criterion, dataloader_val)

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

            # Extracts list of the final AP and AR valus reported
            bbox_stats = coco_evaluator.coco_eval["bbox"].stats

            log.info("\ntrain\t%-10s =  %-15.4f", "AP", bbox_stats[0])
            log.info("train\t%-10s =  %-15.4f", "AP50", bbox_stats[1])
            log.info("train\t%-10s =  %-15.4f", "AP75", bbox_stats[2])
            log.info("train\t%-10s =  %-15.4f", "AP_small", bbox_stats[3])
            log.info("train\t%-10s =  %-15.4f", "AP_medium", bbox_stats[4])
            log.info("train\t%-10s =  %-15.4f", "AP_large", bbox_stats[5])
            log.info("train\t%-10s =  %-15.4f", "AR1", bbox_stats[6])
            log.info("train\t%-10s =  %-15.4f", "AR10", bbox_stats[7])
            log.info("train\t%-10s =  %-15.4f", "AR100", bbox_stats[8])
            log.info("train\t%-10s =  %-15.4f", "AR_small", bbox_stats[9])
            log.info("train\t%-10s =  %-15.4f", "AR_medium", bbox_stats[10])
            log.info("train\t%-10s =  %-15.4f", "AR_large", bbox_stats[11])

            # Current epoch time (train/val)
            one_epoch_time = time.time() - one_epoch_start_time
            print(one_epoch_time)
            one_epoch_time_str = str(datetime.timedelta(seconds=int(one_epoch_time)))
            log.info("\nEpcoh time  (h:mm:ss): %s", one_epoch_time_str)

        # Entire training time
        total_time = time.time() - total_train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info(
            "Training time for %d epochs (h:mm:ss): %s ",
            start_epoch - epochs,
            total_time_str,
        )

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
            breakpoint()
            bbox_predictions = model(samples)

            final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
                bbox_predictions, targets
            )

            if (steps + 1) % self.log_intervals["train_steps_freq"] == 0:
                log.info(
                    "epoch: %-10d iter: %d/%-10d loss: %-10.4f",
                    epoch,
                    steps + 1,
                    len(dataloader_train),
                    final_loss.item(),
                )

            # Calculate gradients and updates weights
            final_loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_val: Iterable,
    ) -> COCOeval:
        """A single forward pass to evluate the val set after training an epoch

        Args:
            model: Model to train
            criterion: Loss function; only used to inspect the loss on the val set,
                       not used for backpropagation
            dataloader_val: Dataloader for the validation set
            device: Device to run the model on
        """

        model.eval()
        
        val_coco_api = convert_to_coco_api(dataloader_val.dataset, bbox_fmt="yolo")
        coco_evaluator = CocoEvaluator(
            val_coco_api, iou_types=["bbox"], bbox_format="coco"
        )

        for steps, (samples, targets) in enumerate(dataloader_val):
            samples = samples.to(self.device)
            targets = [
                {key: value.to(self.device) for key, value in t.items()}
                for t in targets
            ]

            # Inference outputs bbox_preds (tl_x, tl_y, br_x, br_y) and class confidences (num_classes);
            # TODO: This might be wrong comment: these should all be between 0-1 but some look greater than 1, need to investigate
            bbox_preds, class_conf = model(samples, inference=True)

            # TODO, might have to change the output of the bboxes

            # final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
            #    bbox_predictions, targets
            # )

            ## TODO: Comment this
            results = val_preds_to_img_size(samples, targets, bbox_preds, class_conf)

            # TODO: placeholder; change later
            test_stats = 5

            ## TODO: still VERY fuzzy on what the network actually predicts during validation and
            #        how we scale back to original image size
            #breakpoint()
            ### START HERE
            evaluator_time = time.time()

            # results is a dict containing:
            #   "img_id": {boxes: [], "scores": [], "labels", []}
            # where scores is the maximum probability for the class (class probs are mulitplied by objectness probs in an earlier step)
            # and labels is the index of the maximum class probability; reminder
            coco_evaluator.update(results)
            evaluator_time = time.time() - evaluator_time

            if (steps + 1) % self.log_intervals["train_steps_freq"] == 0:
                log.info(
                    "val steps:%d/%-10d ",
                    steps+1,
                    len(dataloader_val),
                )

        coco_evaluator.synchronize_between_processes()

        # Accumulate predictions from all processes
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        return coco_evaluator

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
