import cProfile
import datetime
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import psutil
import torch
from pycocotools.coco import COCO
from torch import nn
from torch.utils import data
from torchvision.transforms import functional as F
from tqdm import tqdm

from detectors import visualize
from detectors.data.coco_eval import CocoEvaluator
from detectors.data.coco_utils import convert_to_coco_api
from detectors.evaluate import evaluate
from detectors.postprocessing.eval import (ap_per_class, get_batch_statistics,
                                           print_eval_stats)
from detectors.postprocessing.nms import non_max_suppression
from detectors.utils import misc
from detectors.utils.box_ops import cxcywh_to_xyxy, val_preds_to_img_size

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
        class_names: List,
        start_epoch: int = 1,
        epochs: int = 100,
        ckpt_epochs: int = 10,
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

        # Visualize the first batch for each dataloader; manually verifies data augmentation correctness
        self._visualize_batch(dataloader_train, "train", class_names)
        self._visualize_batch(dataloader_val, "val", class_names)

        # Starting the epoch at 1 makes calculations more intuitive
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            ## TODO: Implement tensorboard as shown here: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/logger.py#L6

            # Track the time it takes for one epoch (train and val)
            one_epoch_start_time = time.time()

            # Train one epoch
            self._train_one_epoch(
                model,
                criterion,
                dataloader_train,
                optimizer,
                scheduler,
                epoch,
                class_names,
            )

            # Evaluate the model on the validation set
            log.info("\nEvaluating on validation set — epoch %d", epoch)
            metrics_output = self._evaluate(
                model, criterion, dataloader_val, class_names=class_names
            )

            # Save the model every ckpt_epochs
            if (epoch) % ckpt_epochs == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{epoch:04}.pt"
                self._save_model(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_epochs,
                    save_path=ckpt_path,
                )

            # Current epoch time (train/val)
            one_epoch_time = time.time() - one_epoch_start_time
            one_epoch_time_str = str(datetime.timedelta(seconds=int(one_epoch_time)))
            log.info("\nEpoch time  (h:mm:ss): %s", one_epoch_time_str)

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
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        class_names: List[str],
    ):
        """Train one epoch

        Args:
            model: Model to train
            criterion: Loss function
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
            scheduler: Learning rate scheduler to update the learning rate
            epoch: Current epoch; used for logging purposes
        """
        for steps, (samples, targets) in enumerate(dataloader_train):
            samples = samples.to(self.device)
            targets = [
                {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()

            # len(bbox_predictions) = 3; bbox_predictions[i] (B, (5+n_class)*n_bboxes, out_w, out_h)
            bbox_predictions = model(samples)

            final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
                bbox_predictions, targets
            )

            loss_components = misc.to_cpu(
                torch.stack([loss_xy, loss_wh, loss_obj, loss_cls, lossl2])
            )

            # Calculate gradients and updates weights
            final_loss.backward()
            optimizer.step()

            # Calling scheduler step increments a counter which is passed to the lambda function;
            # if .step() is called after every batch, then it will pass the current step;
            # if .step() is called after every epoch, then it will pass the epoch number;
            # this counter is persistent so every epoch it will continue where it left off i.e., it will not reset to 0
            scheduler.step()

            if (steps + 1) % 100 == 0:
                log.info(
                    "Current learning_rate: %s\n",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                )

            if (steps + 1) % self.log_intervals["train_steps_freq"] == 0:
                log.info(
                    "epoch: %-10d iter: %d/%-10d loss: %-10.4f",
                    epoch,
                    steps + 1,
                    len(dataloader_train),
                    final_loss.item(),
                )

                log.info("cpu utilization: %s\n", psutil.virtual_memory().percent)

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_val: Iterable,
        class_names: List,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A single forward pass to evluate the val set after training an epoch

        Args:
            model: Model to train
            criterion: Loss function; only used to inspect the loss on the val set,
                       not used for backpropagation
            dataloader_val: Dataloader for the validation set
            device: Device to run the model on

        Returns:
            A Tuple of the (prec, rec, ap, f1, and class) per class
        """

        # evaluate() is used by both val and test set; this can be customized in the future if needed
        # but for now validation and test behave the same
        metrics_output, detections = evaluate(
            model,
            dataloader_val,
            class_names,
            output_path=self.output_paths["output_dir"],
            device=self.device,
        )

        return metrics_output

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

    def _visualize_batch(
        self, dataloader: data.DataLoader, split: str, class_names: List[str]
    ):
        """Visualize a batch of images after data augmentation; sthis helps manually verify
        the data augmentations are working as intended on the images and boxes

        Args:
            dataloader: Train or val dataloader
            split: "train" or "val"
            class_names: List of class names in the ontology
        """
        valid_splits = {"train", "val"}
        if split not in valid_splits:
            raise ValueError("split must either be in valid_splits")

        samples, targets = next(iter(dataloader))
        visualize.visualize_norm_img_tensors(
            samples,
            targets,
            class_names,
            self.output_paths["output_dir"] / f"{split}-images",
        )
