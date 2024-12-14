import datetime
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils import data

from detectors.evaluate import evaluate, load_model_checkpoint
from detectors.utils import misc
from detectors.visualize import visualize_norm_img_tensors

log = logging.getLogger(__name__)


class Trainer:
    """Trainer TODO: comment"""

    def __init__(
        self,
        output_dir: str,
        device: torch.device = torch.device("cpu"),
        log_train_steps: int = 20,
    ):
        """Constructor for the Trainer class

        Args:
            output_path: Path to save the train outputs
            use_cuda: Whether to use the GPU
        """
        self.device = device

        self.output_dir = Path(output_dir)
        self.log_train_steps = log_train_steps

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: data.DataLoader,
        dataloader_val: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        class_names: list[str],
        start_epoch: int = 1,
        epochs: int = 100,
        ckpt_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
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
        log.info("\ntraining started\n")

        if checkpoint_path is not None:
            start_epoch = load_model_checkpoint(
                checkpoint_path, model, optimizer, self.device, scheduler
            )
            log.info(
                "NOTE: A checkpoint file was provided, the model will resume training at epoch %d",
                start_epoch,
            )

        total_train_start_time = time.time()

        last_best_path = None

        # Visualize the first batch for each dataloader; manually verifies data augmentation correctness
        self._visualize_batch(dataloader_train, "train", class_names)
        self._visualize_batch(dataloader_val, "val", class_names)

        best_ap = 0.0
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
            # TODO: probably save metrics output into csv
            metrics_output, image_detections = self._evaluate(
                model, criterion, dataloader_val, class_names=class_names
            )

            precision, recall, AP, f1, ap_class = metrics_output
            mAP = AP.mean()

            # Save the model every ckpt_epochs
            if (epoch) % ckpt_epochs == 0:
                ckpt_path = self.output_dir / "checkpoints" / f"checkpoint{epoch:04}.pt"
                ckpt_path.parents[0].mkdir(parents=True, exist_ok=True)
                self._save_model(
                    model, optimizer, epoch, save_path=ckpt_path, lr_scheduler=scheduler
                )

            # Save and overwrite the checkpoint with the highest mAP
            if round(mAP, 4) > round(best_ap, 4):
                best_ap = mAP

                mAP_str = f"{mAP*100:.2f}".replace(".", "-")
                best_path = self.output_dir / "checkpoints" / f"best_mAP_{mAP_str}.pt"
                best_path.parents[0].mkdir(parents=True, exist_ok=True)

                log.info(
                    "new best mAP of %.2f found at epoch %d; saving checkpoint",
                    mAP * 100,
                    epoch,
                )
                self._save_model(
                    model, optimizer, epoch, save_path=best_path, lr_scheduler=scheduler
                )

                # delete the previous best mAP model's checkpoint
                if last_best_path is not None:
                    last_best_path.unlink(missing_ok=True)
                last_best_path = best_path

            # Uncomment to visualize validation detections
            # save_dir = self.output_dir / "validation" / f"epoch{epoch}"
            # plot_all_detections(image_detections, classes=class_names, output_dir=save_dir)

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
        for steps, (samples, targets, targets_meta) in enumerate(dataloader_train, 1):
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            # targets = [
            #     {
            #         key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            #         for key, val in t.items()
            #     }
            #     for t in targets
            # ]

            optimizer.zero_grad()

            # list of preds at all 3 scales;
            # bbox_preds[i] (B, (5+n_class)*num_anchors, out_w, out_h)
            bbox_preds = model(samples)

            # final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
            #     bbox_preds, targets, model
            # ) # yolov4
            # loss_components = misc.to_cpu(
            #     torch.stack([loss_xy, loss_wh, loss_obj, loss_cls, lossl2])
            # )

            total_loss, loss_components = criterion(bbox_preds, targets, model)

            # Calculate gradients and updates weights
            total_loss.backward()
            optimizer.step()

            # Calling scheduler step increments a counter which is passed to the lambda function;
            # if .step() is called after every batch, then it will pass the current step;
            # if .step() is called after every epoch, then it will pass the epoch number;
            # this counter is persistent so every epoch it will continue where it left off i.e., it will not reset to 0
            scheduler.step()

            if (steps) % 100 == 0:
                log.info(
                    "Current learning_rate: %s\n",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                )

            if (steps) % self.log_train_steps == 0:
                log.info(
                    "epoch: %-10d iter: %d/%-10d train loss: %-10.4f",
                    epoch,
                    steps,
                    len(dataloader_train),
                    total_loss.item(),
                )

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
            output_path=self.output_dir,
            device=self.device,
        )

        return metrics_output, detections

    def _save_model(
        self,
        model,
        optimizer,
        current_epoch,
        save_path,
        lr_scheduler: Optional[nn.Module] = None,
    ):
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": current_epoch
            + 1,  # + 1 bc when we resume training we want to start at the next step
        }
        if lr_scheduler is not None:
            save_dict["lr_scheduler"] = lr_scheduler.state_dict()

        torch.save(
            save_dict,
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

        samples, targets, annoations = next(iter(dataloader))
        visualize_norm_img_tensors(
            samples,
            targets,
            class_names,
            self.output_dir / "aug" / f"{split}-images",
            annoations,
        )
